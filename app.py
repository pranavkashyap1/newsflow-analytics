import streamlit as st
import os

st.set_page_config(
    page_title="NewsFlow Analytics",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-setup on first run
if not os.path.exists("data/processed/articles.csv") or not os.path.exists("models/rf_regressor.pkl"):
    with st.spinner("Setting up NewsFlow Analytics for first run — this takes about 60 seconds..."):
        from setup import setup
        setup()
    st.success("Setup complete. Reloading...")
    st.rerun()

st.title("📰 NewsFlow Analytics")
st.subheader("Media Traffic Intelligence Platform")

st.markdown("""
Welcome to **NewsFlow Analytics** — an AI-powered platform for editorial intelligence.

Use the sidebar to navigate:

| Page | Description |
|---|---|
| 📊 Executive | KPI overview and top-level metrics |
| 📈 Analytics | Deep-dive category and trend analysis |
| 🤖 ML Predictor | Headline performance prediction |
| 💡 AI Assistant | AI-powered editorial recommendations |
| 📄 Reports | Automated PDF report generation |
""")

st.info("👈 Select a page from the sidebar to get started.")
