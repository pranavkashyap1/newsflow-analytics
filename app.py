import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="NewsFlow Analytics",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0f1117;
}
.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    padding: 10px 0 4px 0;
    letter-spacing: 0.5px;
}
.sidebar-sub {
    font-size: 12px;
    color: #888;
    margin-bottom: 20px;
}
.kpi-card {
    background: linear-gradient(135deg, #1e2130, #252836);
    border-radius: 12px;
    padding: 20px 24px;
    border-left: 4px solid #4CAF50;
    margin-bottom: 8px;
}
.kpi-card.blue  { border-left-color: #2196F3; }
.kpi-card.orange{ border-left-color: #FF9800; }
.kpi-card.purple{ border-left-color: #9C27B0; }
.kpi-card.green { border-left-color: #4CAF50; }
.kpi-label {
    font-size: 13px;
    color: #aaa;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
}
.hero-title {
    font-size: 48px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.2;
    margin-bottom: 16px;
}
.hero-sub {
    font-size: 18px;
    color: #aaaaaa;
    margin-bottom: 32px;
    line-height: 1.6;
}
.step-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    height: 160px;
    border: 1px solid #2a2d3e;
}
.step-number {
    font-size: 32px;
    font-weight: 800;
    color: #2196F3;
    margin-bottom: 8px;
}
.step-title {
    font-size: 15px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 6px;
}
.step-desc {
    font-size: 12px;
    color: #888;
}
.feature-badge {
    display: inline-block;
    background: #1e2130;
    border: 1px solid #2a2d3e;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 13px;
    color: #ccc;
    margin: 4px;
}
.divider {
    border: none;
    border-top: 1px solid #2a2d3e;
    margin: 32px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar branding ──
with st.sidebar:
    st.markdown('<div class="sidebar-title">📰 NewsFlow Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Media Traffic Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Navigate**")

# ── Auto setup ──
if not os.path.exists("data/processed/articles.csv") or not os.path.exists("models/rf_regressor.pkl"):
    with st.spinner("Setting up NewsFlow Analytics for first run — this takes about 60 seconds..."):
        from setup import setup
        setup()
    st.success("Setup complete.")
    st.rerun()

# ── Load data for live KPIs ──
@st.cache_data
def load_kpis():
    from src.analytics import get_kpis, category_performance
    df = pd.read_csv("data/processed/articles.csv", quoting=1)
    return get_kpis(df), category_performance(df), len(df)

kpis, cat_df, total = load_kpis()

# ── Hero Section ──
st.markdown('<div class="hero-title">📰 NewsFlow Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">AI-powered media intelligence platform that helps digital newsrooms '
    'analyze content performance, predict article engagement, and automate editorial reporting.</div>',
    unsafe_allow_html=True
)

# Target audience badge
st.markdown("""
<div>
<span class="feature-badge">🏢 Built for Media Companies</span>
<span class="feature-badge">🤖 AI + Machine Learning</span>
<span class="feature-badge">📊 Real-time Analytics</span>
<span class="feature-badge">📄 Automated Reporting</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Live KPI Cards ──
st.markdown("### 📈 Platform Overview — Live Data")
st.caption("Powered by 120,000 AG News articles with engineered engagement metrics")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="kpi-card blue">
        <div class="kpi-label">Total Articles</div>
        <div class="kpi-value">{kpis['total_articles']:,}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-card green">
        <div class="kpi-label">Total Views</div>
        <div class="kpi-value">{kpis['total_views'] / 1_000_000:.1f}M</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="kpi-card orange">
        <div class="kpi-label">Top Category</div>
        <div class="kpi-value">{kpis['top_category']}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="kpi-card purple">
        <div class="kpi-label">Avg Engagement</div>
        <div class="kpi-value">{kpis['avg_engagement']:,.0f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── How It Works ──
st.markdown("### 🔄 How It Works")
st.caption("Four steps from raw data to editorial intelligence")

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">01</div>
        <div class="step-title">Data Pipeline</div>
        <div class="step-desc">120K news articles loaded, cleaned and enriched with 19 engineered features</div>
    </div>""", unsafe_allow_html=True)
with s2:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">02</div>
        <div class="step-title">ML Models</div>
        <div class="step-desc">Random Forest + TF-IDF predict engagement scores and top performer probability</div>
    </div>""", unsafe_allow_html=True)
with s3:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">03</div>
        <div class="step-title">BI Dashboard</div>
        <div class="step-desc">Interactive Plotly charts surface insights across categories, trends and authors</div>
    </div>""", unsafe_allow_html=True)
with s4:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">04</div>
        <div class="step-title">AI + Reports</div>
        <div class="step-desc">Groq LLaMA generates editorial recommendations. One-click PDF report export</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Navigation Guide ──
st.markdown("### 🧭 Where to Go")
st.caption("Use the sidebar to navigate — here's what each page does")

n1, n2, n3 = st.columns(3)
with n1:
    st.info("**📊 Executive Dashboard**\nTop-level KPIs, category distribution, and top performing articles. Start here for the big picture.")
    st.info("**📈 Analytics Dashboard**\nWeekly trends, engagement distribution, author rankings. Filter by category.")
with n2:
    st.success("**🤖 ML Predictor**\nType any headline and get instant engagement prediction, category classification, and editorial insights.")
    st.success("**💡 AI Assistant**\nGroq-powered editorial strategy recommendations and AI headline generator.")
with n3:
    st.warning("**📄 Reports**\nGenerate and download a complete weekly performance report as PDF in one click.")
    st.markdown("""
    <div class="step-card" style="height:auto; text-align:left; padding:16px;">
        <div style="font-size:13px; color:#aaa; margin-bottom:8px;">TECH STACK</div>
        <div style="font-size:12px; color:#ccc; line-height:2;">
        🐍 Python 3.11<br>
        📊 Streamlit + Plotly<br>
        🤖 Scikit-learn + TF-IDF<br>
        💡 Groq LLaMA 3.1<br>
        📄 FPDF2<br>
        📦 AG News Dataset
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.caption("NewsFlow Analytics — Built for Times Group Internship | Data Science & AI Automation")
