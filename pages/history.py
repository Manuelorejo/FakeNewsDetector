import streamlit as st
import sqlite3
from datetime import datetime

# ------------------------------
# LOCK PAGE UNTIL LOGIN
# ------------------------------
if "user_id" not in st.session_state or st.session_state.user_id is None:
    st.warning("⚠️ Please log in first! Go to the Login page.")
    st.stop()

DB_PATH = "app_data.db"

def get_user_history(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT title, url, verdict, satire_prob, fake_prob, timestamp 
        FROM history 
        WHERE user_id = ? 
        ORDER BY id DESC
    """, (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

st.title("🕘 Your Analysis History")
history = get_user_history(st.session_state.user_id)

if not history:
    st.info("You haven’t analyzed any articles yet.")
else:
    for i, entry in enumerate(history, 1):
        title, url, verdict, satire_prob, fake_prob, timestamp = entry
        st.markdown(f"**{i}. {title}**")
        st.markdown(f"- URL: {url or 'N/A'}")
        st.markdown(f"- Verdict: {verdict.capitalize()}")
        st.markdown(f"- Satire Probability: {satire_prob:.0%}")
        st.markdown(f"- Fake Probability: {fake_prob:.0%}")
        st.markdown(f"- Analyzed at: {timestamp}")
        st.markdown("---")
