import streamlit as st
import os

st.title("Debug — API Key Status")

st.subheader("st.secrets")
try:
    keys = list(st.secrets.keys())
    st.write(f"Keys in st.secrets: {keys}")
    if "GROQ_API_KEY" in st.secrets:
        val = st.secrets["GROQ_API_KEY"]
        st.success(f"GROQ_API_KEY found in st.secrets, length={len(val)}")
    else:
        st.error("GROQ_API_KEY NOT in st.secrets")
except Exception as e:
    st.error(f"st.secrets error: {e}")

st.subheader("os.environ")
env_key = os.getenv("GROQ_API_KEY")
if env_key:
    st.success(f"GROQ_API_KEY found in os.environ, length={len(env_key)}")
else:
    st.error("GROQ_API_KEY NOT in os.environ")
