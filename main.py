# -*- coding: utf-8 -*-
"""
Main Fake News Detector Page
Locked until login
"""

import streamlit as st
import requests
import joblib
import numpy as np
from urllib.parse import urlparse
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
from db import add_history
from db import init_db


init_db()
# ------------------------------
# LOCK PAGE UNTIL LOGIN
# ------------------------------

def logout():
    st.session_state.user_id = None
    st.session_state.username = None
    st.success("You have been logged out.")
    st.switch_page("pages/login.py")

with st.sidebar:
    st.markdown("### 👤 Account")
    st.write(f"Logged in as **{st.session_state.get('username','')}**")
    if st.button("🚪 Log out"):
        logout()
        
if "user_id" not in st.session_state or st.session_state.user_id is None:
    st.warning("⚠️ Please log in first.")
    st.switch_page("pages/login.py")

st.title("📰 Fake News Detector")
st.success(f"✅ Welcome! You are now logged in.")


# ------------------------------
# IMPORT SCRAPERS
# ------------------------------
from bbc import scrape_bbc_article
from pulse_ng import scrape_pulse_article
from punch import scrape_punch_article
from instablog import scrape_instablog_article
from onion import scrape_onion_article
from fox import scrape_fox_article

SCRAPER_MAP = {
    "bbc.com": scrape_bbc_article,
    "www.pulse.ng": scrape_pulse_article,
    "pulse.ng": scrape_pulse_article,
    "punchng.com": scrape_punch_article,
    "instablog9ja.com": scrape_instablog_article,
    "theonion.com": scrape_onion_article,
    "foxnews.com": scrape_fox_article
}

# ------------------------------
# LOAD MODELS
# ------------------------------
@st.cache_resource
def load_models():
    return (
        joblib.load("model.pkl"),
        joblib.load("vectorizer.pkl"),
        joblib.load("Satire_model.pkl"),
        joblib.load("Satire_vectorizer.pkl")
    )

model, vectorizer, satire_model, satire_vectorizer = load_models()

SATIRE_HIGH = 0.70
SATIRE_LOW = 0.40
FAKE_HIGH = 0.75
FAKE_UNCERTAIN = 0.55

# ------------------------------
# UTILITY FUNCTIONS
# ------------------------------
def predict_satire_prob(title, text):
    X = satire_vectorizer.transform([f"{title}. {text}"])
    if hasattr(satire_model, "predict_proba"):
        return satire_model.predict_proba(X)[0][1]
    else:
        score = satire_model.decision_function(X)[0]
        return 1 / (1 + np.exp(-score))

def predict_fake_prob(title, text):
    X = vectorizer.transform([f"{title}. {text}"])
    return model.predict_proba(X)[0][1]

def plot_probability_pie(satire_prob, fake_prob):
    credible_prob = max(0, 1 - satire_prob - fake_prob)
    labels = ["Satire", "Fake", "Credible"]
    values = [satire_prob, fake_prob, credible_prob]
    colors = ["#FF6B6B","#FFCA3A","#4CAF50"]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3, marker_colors=colors)])
    fig.update_layout(showlegend=True, margin=dict(t=0,b=0,l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)

def timeline_step(title, status, description=""):
    colors = {"pass":"#28a745","warn":"#ffc107","fail":"#dc3545","pending":"#6c757d"}
    st.markdown(f"""
        <div style='padding:15px 20px;border-left:6px solid {colors.get(status,"#aaa")};margin-bottom:10px;border-radius:6px;background-color:#f8f9fa;'>
            <strong>{title}</strong><br>{description}
        </div>
    """, unsafe_allow_html=True)

# ------------------------------
# SQLITE HISTORY FUNCTIONS
# ------------------------------
DB_PATH = "app_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            url TEXT,
            title TEXT NOT NULL,
            verdict TEXT NOT NULL,
            satire_prob REAL,
            fake_prob REAL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_to_history(user_id, url, title, verdict, satire_prob, fake_prob):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO history (user_id, url, title, verdict, satire_prob, fake_prob, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, url, title, verdict, satire_prob, fake_prob, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

init_db()

# ------------------------------
# MAIN INPUT & SCRAPER STATUS
# ------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    url_input = st.text_input("Article URL", placeholder="Paste article URL here...")
    title_input = st.text_input("Headline (manual input)")
    text_input = st.text_area("Article text (manual input)", height=200)

with col2:
    scraper_status_placeholder = st.empty()  # Right-side feedback box

# ------------------------------
# ANALYZE BUTTON
# ------------------------------
if st.button("Analyze"):
    article_title = title_input.strip()
    article_text = text_input.strip()

    # --- Scrape if URL provided ---
    if url_input.strip():
        domain = urlparse(url_input).netloc.replace("www.","")
        scraper = SCRAPER_MAP.get(domain)

        if not scraper:
            scraper_status_placeholder.error("❌ Website not supported for scraping.")
            st.stop()

        try:
            with st.spinner("🔍 Scraping article..."):
                data = scraper(url_input)
            article_title = data.get("title", article_title)
            article_text = data.get("text", article_text)

            scraper_status_placeholder.success(f"✅ Article detected!\n\n*{article_title}*")

        except Exception as e:
            scraper_status_placeholder.error(f"❌ Scraping failed: {e}")
            st.stop()
    else:
        scraper_status_placeholder.info("ℹ️ No URL provided. Using manual input.")

    if not article_title and not article_text:
        st.warning("Please provide headline or article text.")
        st.stop()

    # ------------------------------
    # ANALYSIS WORKFLOW
    # ------------------------------
    st.markdown("## 🧭 Analysis Timeline")
    verdict = None
    satire_warn = False

    # -------- Satire Detection --------
    satire_prob = predict_satire_prob(article_title, article_text)
    if url_input.strip() and "theonion.com" in url_input:
        satire_prob = min(1.0, satire_prob + 0.6)

    if satire_prob >= SATIRE_HIGH:
        timeline_step("Satire Detection", "fail", f"High satire detected ({satire_prob:.2%})")
        verdict = "satire"
    elif SATIRE_LOW <= satire_prob < SATIRE_HIGH:
        timeline_step("Satire Detection", "warn", f"Moderate satire ({satire_prob:.2%})")
        satire_warn = True
    else:
        timeline_step("Satire Detection", "pass", f"Low satire ({satire_prob:.2%})")

    # -------- Credibility --------
    fake_prob = predict_fake_prob(article_title, article_text)
    final_fake_prob = fake_prob
    if url_input.strip() and "theonion.com" in url_input:
        final_fake_prob = max(0, fake_prob - 0.2)

    if verdict is None:
        if final_fake_prob >= FAKE_HIGH:
            timeline_step("Credibility", "fail", f"High likelihood of misinformation ({final_fake_prob:.2%})")
            verdict = "fake"
        elif FAKE_UNCERTAIN <= final_fake_prob < FAKE_HIGH:
            timeline_step("Credibility", "warn", f"Inconclusive result ({final_fake_prob:.2%})")
            verdict = "unverified"
        else:
            timeline_step("Credibility", "pass", f"Likely credible ({1-final_fake_prob:.2%})")
            verdict = "real"

    # -------- Pie Chart Explainability --------
    st.markdown("## 📊 Model Explainability")
    plot_probability_pie(satire_prob, final_fake_prob)

    # -------- Final Verdict --------
    verdict_text = {
        "satire":"Likely Satirical",
        "fake":"Likely Misinformation",
        "unverified":"Unverified",
        "real":"Likely Credible"
    }
    verdict_colors = {
        "satire":"#FF6B6B","fake":"#DC3545","unverified":"#FFC107","real":"#4CAF50"
    }
    st.markdown(f"<div style='padding:20px;border-radius:16px;background-color:{verdict_colors.get(verdict,'#EEE')};font-weight:bold;text-align:center;'>{verdict_text.get(verdict,'Unknown')}</div>", unsafe_allow_html=True)

    if satire_warn and verdict != "satire":
        st.info("⚠️ Moderate satirical elements detected — content may include exaggeration or humor.")

    # -------- Save to history --------
   
    
    add_history(st.session_state.user_id, url_input, article_title, verdict, satire_prob, final_fake_prob)

