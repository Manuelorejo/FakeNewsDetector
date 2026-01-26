import streamlit as st
from db import validate_user, init_db

init_db()

st.title("🔐 Login")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    user_id = validate_user(username, password)

    if user_id:
        # Set auth state
        st.session_state.user_id = user_id
        st.session_state.username = username

        # Success message (brief)
        st.success("✅ Login successful! Redirecting...")

        # 🚀 ACTUAL REDIRECT
        st.switch_page("main.py")

    else:
        st.error("❌ Invalid username or password")

st.markdown("Don't have an account?")

if st.button("👉 Sign up here"):
    st.switch_page("pages/register.py")



