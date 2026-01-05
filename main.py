import streamlit as st
import requests
import joblib
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f9f9f9;
    }
    .main-container {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.85rem;
        font-size: 1.1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .status-pass { color: #28a745; font-weight: bold; }
    .status-warn { color: #ffc107; font-weight: bold; }
    .status-fail { color: #dc3545; font-weight: bold; }
    .status-pending { color: #6c757d; }
    .verdict-box {
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .verdict-real { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .verdict-fake { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    .verdict-unverified { background-color: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }
    .verdict-satire { background-color: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }
    .verdict-factchecked { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    footer { 
        text-align: center; 
        padding: 1.5rem; 
        color: #666; 
        font-size: 0.9rem; 
        border-top: 1px solid #eee; 
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GOOGLE FACT CHECK API (KEY HIDDEN)
# ─────────────────────────────────────────────
# IMPORTANT: Set your Google Fact Check Tools API key as an environment variable
# In your terminal or deployment platform: export GOOGLE_API_KEY="your_key_here"
# Or in Streamlit Secrets: add GOOGLE_API_KEY in secrets.toml
import os
API_KEY = os.getenv("GOOGLE_API_KEY")  # Securely loaded from environment
if not API_KEY:
    API_KEY = ""  # Will gracefully disable fact-checking if missing

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def google_fact_check(query, max_results=3):
    if not API_KEY:
        return []  # Skip fact-check if no key
    params = {
        "query": query,
        "key": API_KEY,
        "pageSize": max_results
    }
    try:
        res = requests.get(API_URL, params=params, timeout=10)
        if res.status_code != 200:
            return []
        return res.json().get("claims", [])
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
# THRESHOLDS
# ─────────────────────────────────────────────
SATIRE_HIGH = 0.70
SATIRE_LOW = 0.40
FAKE_HIGH = 0.75
FAKE_UNCERTAIN = 0.55

# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def confidence_bar(prob, label="Confidence"):
    color = "#dc3545" if prob > 0.7 else "#ffc107" if prob > 0.4 else "#28a745"
    st.markdown(f"""
        <div style="background-color: #e9ecef; border-radius: 8px; height: 24px; overflow: hidden; margin: 10px 0;">
            <div style="background-color: {color}; width: {prob*100}%; height: 100%; transition: width 0.8s ease;"></div>
        </div>
        <p style="text-align: center; color: #555; margin-top: -5px;">{label}: {prob:.2%}</p>
    """, unsafe_allow_html=True)

def timeline_step(title, status, description=None):
    icons = {"pending": "🕐", "pass": "✅", "warn": "⚠️", "fail": "🚫"}
    st.markdown(
        f"""
        <div style="padding:15px 0; border-left: 4px solid {'#28a745' if status=='pass' else '#ffc107' if status=='warn' else '#dc3545' if status=='fail' else '#aaa'}; padding-left: 20px; margin-bottom: 10px;">
            <h4 style="margin:0; color:{'#28a745' if status=='pass' else '#ffc107' if status=='warn' else '#dc3545' if status=='fail' else '#666'}">
                {icons[status]} {title}
            </h4>
            <div style="color:#666; margin-top:5px;">
                {description or ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# MODEL FUNCTIONS
# ─────────────────────────────────────────────
def predict_satire_prob(title, text):
    combined = f"{title}. {text}"
    X = satire_vectorizer.transform([combined])
    if hasattr(satire_model, "predict_proba"):
        return satire_model.predict_proba(X)[0][1]
    else:
        score = satire_model.decision_function(X)[0]
        return 1 / (1 + np.exp(-score))

def predict_fake_prob(title, text):
    combined = f"{title}. {text}"
    X = vectorizer.transform([combined])
    return model.predict_proba(X)[0][1]

# ─────────────────────────────────────────────
# MAIN APP UI
# ─────────────────────────────────────────────
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.title("📰 Fake News Detection Platform")
st.caption("Multi-layer analysis: Fact-Checking → Satire Detection → ML Credibility Assessment")

col1, col2 = st.columns([3, 1])
with col1:
    title_input = st.text_input("News Headline", placeholder="Enter the headline...")
    text_input = st.text_area("News Article Text", height=280, placeholder="Paste the full article here...")

st.markdown("</div>", unsafe_allow_html=True)

if st.button("🔍 Analyze News"):
    if not title_input.strip() and not text_input.strip():
        st.warning("Please provide a headline or article text to analyze.")
        st.stop()

    st.markdown("## 🧭 Analysis Timeline")
    st.caption("Step-by-step evaluation process")

    verdict = None
    satire_warn = False

    # Step 1: Fact Check
    with st.spinner("Searching fact-check databases..."):
        claims = google_fact_check(title_input)

    if claims:
        timeline_step("Step 1: Fact-Check Databases", "fail",
                      "This claim has been previously reviewed by professional fact-checkers.")
        with st.expander("🔍 View Fact-Check Details", expanded=True):
            for claim in claims:
                st.markdown(f"**Claim:** {claim.get('text', 'N/A')}")
                for review in claim.get("claimReview", []):
                    publisher = review.get('publisher', {}).get('name', 'Unknown')
                    rating = review.get('textualRating', 'N/A')
                    url = review.get('url')
                    st.markdown(f"- **{publisher}**: `{rating}`")
                    if url:
                        st.link_button("View Full Report →", url)
        verdict = "factchecked"
    else:
        timeline_step("Step 1: Fact-Check Databases", "pass",
                      "No prior fact-checks found. Proceeding to next stages.")

    if verdict is None:
        # Step 2: Satire Detection
        with st.spinner("Detecting satirical content..."):
            satire_prob = predict_satire_prob(title_input, text_input)

        if satire_prob >= SATIRE_HIGH:
            timeline_step("Step 2: Satire Detection", "fail",
                          f"Strong satirical indicators detected ({satire_prob:.2%}).")
            confidence_bar(satire_prob, "Satire Confidence")
            verdict = "satire"
        elif SATIRE_LOW <= satire_prob < SATIRE_HIGH:
            timeline_step("Step 2: Satire Detection", "warn",
                          f"Moderate satire signals ({satire_prob:.2%}). Proceeding with caution.")
            confidence_bar(satire_prob, "Satire Confidence")
            satire_warn = True
        else:
            timeline_step("Step 2: Satire Detection", "pass",
                          f"Low satire likelihood ({satire_prob:.2%}).")
            confidence_bar(satire_prob, "Satire Confidence")

    if verdict is None:
        # Step 3: ML Credibility
        with st.spinner("Running credibility assessment..."):
            fake_prob = predict_fake_prob(title_input, text_input)

        if fake_prob >= FAKE_HIGH:
            timeline_step("Step 3: Credibility Assessment", "fail",
                          f"High likelihood of misinformation ({fake_prob:.2%}).")
            confidence_bar(fake_prob, "Misinformation Probability")
            verdict = "fake"
        elif FAKE_UNCERTAIN <= fake_prob < FAKE_HIGH:
            timeline_step("Step 3: Credibility Assessment", "warn",
                          f"Inconclusive result ({fake_prob:.2%}). Manual verification advised.")
            confidence_bar(fake_prob, "Misinformation Probability")
            verdict = "unverified"
        else:
            timeline_step("Step 3: Credibility Assessment", "pass",
                          f"Likely credible news ({1 - fake_prob:.2%} confidence).")
            confidence_bar(1 - fake_prob, "Credibility Confidence")
            verdict = "real"

    # Final Verdict Box
    st.markdown("## 📋 Final Verdict")
    verdict_text = {
        "factchecked": "Previously Fact-Checked",
        "satire": "Likely Satirical Content",
        "fake": "Likely Misinformation",
        "unverified": "Unverified / Inconclusive",
        "real": "Likely Credible"
    }
    verdict_class = {
        "factchecked": "verdict-factchecked",
        "satire": "verdict-satire",
        "fake": "verdict-fake",
        "unverified": "verdict-unverified",
        "real": "verdict-real"
    }

    st.markdown(f"""
        <div class="verdict-box {verdict_class[verdict]}">
            {verdict_text[verdict]}
        </div>
    """, unsafe_allow_html=True)

    if satire_warn and verdict != "satire":
        st.info("ℹ️ Moderate satirical elements detected — content may include exaggeration or humor.")

    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        - **Layer 1**: Queries professional fact-checking sources first.
        - **Layer 2**: Specialized model detects satire and humor.
        - **Layer 3**: General machine learning classifier evaluates credibility patterns.
        - Confidence thresholds prevent overconfident or incorrect labels.
        - Always cross-verify important information from multiple trusted sources.
        """)

# ─────────────────────────────────────────────
# FOOTER (replaces sidebar)
# ─────────────────────────────────────────────
st.markdown("""
    <footer>
        <p><strong>Fake News Detection Platform</strong> • Multi-layer verification tool for demonstration purposes</p>
        <p>
            Fact-checking powered by Google Fact Check Tools (API key required via environment variable) • 
            Models trained on public datasets • Not infallible — use critical thinking
        </p>
        <p>Thresholds: Satire (High >70%), Fake (High >75%, Uncertain 55–75%)</p>
    </footer>
    """, unsafe_allow_html=True)