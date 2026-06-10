
import streamlit as st

st.set_page_config(
    page_title="NewsFlow Analytics",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
| 💡 AI Assistant | Gemini-powered editorial recommendations |
| 📄 Reports | Automated PDF report generation |
""")

st.info("👈 Select a page from the sidebar to get started.")
