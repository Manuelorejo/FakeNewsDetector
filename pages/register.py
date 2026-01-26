import streamlit as st
import time
from db import add_user
from db import init_db

init_db()
st.title("📝 Register")

username = st.text_input("Username")
password = st.text_input("Password", type="password")
confirm_password = st.text_input("Confirm Password", type="password")

if st.button("Sign Up"):
    if not username or not password or not confirm_password:
        st.warning("⚠️ Fill all fields")
    elif password != confirm_password:
        st.error("❌ Passwords do not match")
    else:
        success = add_user(username, password)
        if success:
            st.success("✅ Account created! Redirecting to login...")
            time.sleep(3)
            st.switch_page("pages/login.py")


        else:
            st.error("❌ Username already exists. Pick another.")


