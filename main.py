import streamlit as st
import requests
import joblib
import numpy as np
from urllib.parse import urlparse
import os

# ─────────────────────────────────────────────
# IMPORT SCRAPERS
# ─────────────────────────────────────────────
from bbc import scrape_bbc_article
from instablog import scrape_instablog_article
from onion import scrape_onion_article
from pulse_ng import scrape_pulse_article
from punch import scrape_punch_article

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detection Platform")

# ─────────────────────────────────────────────
# GOOGLE FACT CHECK API
# ─────────────────────────────────────────────
API_KEY = os.getenv("GOOGLE_API_KEY", "")
API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def google_fact_check(query, max_results=3):
    if not API_KEY:
        return []
    try:
        res = requests.get(
            API_URL,
            params={"query": query, "key": API_KEY, "pageSize": max_results},
            timeout=10
        )
        return res.json().get("claims", []) if res.status_code == 200 else []
    except Exception:
        return []

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    return (
        joblib.load("model.pkl"),
        joblib.load("vectorizer.pkl"),
        joblib.load("Satire_model.pkl"),
        joblib.load("Satire_vectorizer.pkl")
    )

model, vectorizer, satire_model, satire_vectorizer = load_models()

# ─────────────────────────────────────────────
# SOURCE WEIGHTS / SATIRE BOOST
# ─────────────────────────────────────────────
SOURCE_WEIGHTS = {
    "bbc.com":          {"name": "BBC News",      "credibility": 0.04,  "satire_bias": 0.0},
    "pulse.ng":         {"name": "Pulse NG",      "credibility": 0.02,  "satire_bias": 0.0},
    "www.pulse.ng":     {"name": "Pulse NG",      "credibility": 0.02,  "satire_bias": 0.0},
    "punchng.com":      {"name": "Punch Nigeria", "credibility": 0.01,  "satire_bias": 0.0},
    "instablog9ja.com": {"name": "Instablog9ja",  "credibility": -0.02, "satire_bias": 0.0},
    "theonion.com":     {"name": "The Onion",     "credibility": 0.0,   "satire_bias": 0.30},
}

# ─────────────────────────────────────────────
# SCRAPER ROUTER
# ─────────────────────────────────────────────
def scrape_article_from_url(url):
    domain = urlparse(url).netloc.lower()
    scraper_map = {
        "bbc.com": scrape_bbc_article,
        "instablog9ja.com": scrape_instablog_article,
        "theonion.com": scrape_onion_article,
        "pulse.ng": scrape_pulse_article,
        "www.pulse.ng": scrape_pulse_article,
        "punchng.com": scrape_punch_article,
    }

    for key, scraper in scraper_map.items():
        if key in domain:
            try:
                data = scraper(url)
                # Convert tuple to dict if needed
                if isinstance(data, tuple):
                    data = {"title": data[0], "text": data[1]}
                data["source_name"] = SOURCE_WEIGHTS[key]["name"]
                return data
            except Exception as e:
                raise RuntimeError(f"Scraper failed: {e}")

    raise ValueError(
        "Unsupported news source. Only BBC, Pulse NG, Punch, Instablog9ja, and The Onion are supported."
    )

# ─────────────────────────────────────────────
# MODEL FUNCTIONS
# ─────────────────────────────────────────────
def predict_fake_prob(title, text, source_domain=None):
    combined = f"{title}. {text}"
    X = vectorizer.transform([combined])
    base_prob = model.predict_proba(X)[0][1]
    bias = SOURCE_WEIGHTS.get(source_domain, {}).get("credibility", 0.0)
    adjusted_prob = min(max(base_prob - bias, 0.0), 1.0)
    return adjusted_prob

def predict_satire_prob(title, text, source_domain=None):
    combined = f"{title}. {text}"
    X = satire_vectorizer.transform([combined])
    if hasattr(satire_model, "predict_proba"):
        base_prob = satire_model.predict_proba(X)[0][1]
    else:
        base_prob = 1 / (1 + np.exp(-satire_model.decision_function(X)[0]))
    boost = SOURCE_WEIGHTS.get(source_domain, {}).get("satire_bias", 0.0)
    adjusted_prob = min(base_prob + boost * (1 - base_prob), 1.0)
    return adjusted_prob

def explain_prediction(text):
    features = vectorizer.get_feature_names_out()
    tfidf = vectorizer.transform([text]).toarray()[0]
    top = np.argsort(tfidf)[-6:][::-1]
    return [features[i] for i in top if tfidf[i] > 0]

def confidence_bar(prob, label="Confidence"):
    color = "#dc3545" if prob > 0.7 else "#ffc107" if prob > 0.4 else "#28a745"
    st.markdown(f"""
        <div style="background-color: #e9ecef; border-radius: 8px; height: 24px; overflow: hidden; margin: 10px 0;">
            <div style="background-color: {color}; width: {prob*100}%; height: 100%; transition: width 0.8s ease;"></div>
        </div>
        <p style="text-align: center; color: #555; margin-top: -5px;">{label}: {prob:.2%}</p>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────
url_input = st.text_input("Optional: Paste a news article URL")
title_input = st.text_input("News Headline")
text_input = st.text_area("News Article Text", height=260)
source_display = st.empty()

if st.button("🔍 Analyze"):
    source_domain = None

    # --- URL SCRAPING ---
    if url_input.strip():
        try:
            article = scrape_article_from_url(url_input)
            title_input = article["title"]
            text_input = article["text"]
            source_display.caption(f"Detected source: {article.get('source_name', 'Unknown')}")
            source_domain = urlparse(url_input).netloc.lower()
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error("Failed to scrape article.")
            st.caption(f"Error details: {e}")

    # --- VALIDATE INPUT ---
    if not title_input.strip() or not text_input.strip():
        st.warning("Please provide a headline and article text.")
        st.stop()

    # --- FACT CHECK ---
    claims = google_fact_check(title_input)
    if claims:
        st.error("This claim has already been fact-checked.")
        st.stop()

    # --- PREDICTIONS ---
    satire_prob = predict_satire_prob(title_input, text_input, source_domain)
    fake_prob = predict_fake_prob(title_input, text_input, source_domain)

    # --- VERDICT LOGIC ---
    if satire_prob > 0.7:
        verdict = "Satire"
    elif fake_prob > 0.75:
        verdict = "Likely Misinformation"
    elif fake_prob > 0.55:
        verdict = "Unverified"
    else:
        verdict = "Likely Credible"

    st.markdown(f"## 🧠 Verdict: **{verdict}**")

    # --- CONFIDENCE BARS ---
    confidence_bar(fake_prob, "Misinformation Probability")
    confidence_bar(satire_prob, "Satire Probability")

    # --- MODEL EXPLAINABILITY ---
    with st.expander("🔎 Model Explanation"):
        st.markdown(f"**Fake news probability:** {fake_prob:.2%}")
        st.markdown(f"**Satire probability:** {satire_prob:.2%}")
        st.markdown("**Top influencing words:**")
        for word in explain_prediction(text_input):
            st.markdown(f"- `{word}`")
