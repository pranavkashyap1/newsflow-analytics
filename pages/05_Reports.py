import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.report_generator import generate_report
from src.analytics import get_kpis, category_performance, top_articles

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")

st.markdown("""
<style>
.kpi-card {
    background: linear-gradient(135deg, #1e2130, #252836);
    border-radius: 12px;
    padding: 20px 24px;
    border-left: 4px solid #4CAF50;
    margin-bottom: 8px;
}
.kpi-card.blue   { border-left-color: #2196F3; }
.kpi-card.green  { border-left-color: #4CAF50; }
.kpi-card.orange { border-left-color: #FF9800; }
.kpi-card.purple { border-left-color: #9C27B0; }
.kpi-label {
    font-size: 12px;
    color: #aaa;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
}
.report-section {
    background: #1e2130;
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid #2a2d3e;
    margin: 8px 0;
}
.report-section-title {
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 6px;
}
.report-section-desc {
    font-size: 13px;
    color: #aaa;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/articles.csv", quoting=1)

df     = load_data()
kpis   = get_kpis(df)
cat_df = category_performance(df)
top_df = top_articles(df, n=5)

# ── Header ──
st.markdown("## 📄 Automated Report Generation")
st.caption("Generate a complete editorial performance report as a professional PDF — ready to share with editorial leadership")
st.divider()

# ── What's in the report ──
st.markdown("#### What's Included in the Report")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">📊 Executive KPIs</div>
        <div class="report-section-desc">Total articles, views, engagement scores, top category and top author — the numbers leadership cares about</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">📋 Category Breakdown</div>
        <div class="report-section-desc">Performance table comparing World, Sports, Business and Sci/Tech across all metrics</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">🏆 Top 5 Articles</div>
        <div class="report-section-desc">Best performing articles by engagement score with category, views and author attribution</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">💡 Editorial Recommendations</div>
        <div class="report-section-desc">5 specific, data-driven recommendations for the editorial team based on current performance</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">📅 Report Period</div>
        <div class="report-section-desc">Automatically calculated from your dataset date range — no manual input needed</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">🏢 Professional Format</div>
        <div class="report-section-desc">Clean PDF layout with headers, tables, and branded footer — ready to share with stakeholders</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Live preview ──
st.markdown("#### Live Data Preview")
st.caption("This is exactly what will appear in the generated report")

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Total Articles</div><div class="kpi-value">{kpis["total_articles"]:,}</div></div>', unsafe_allow_html=True)
with r2:
    st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Total Views</div><div class="kpi-value">{kpis["total_views"]/1_000_000:.1f}M</div></div>', unsafe_allow_html=True)
with r3:
    st.markdown(f'<div class="kpi-card orange"><div class="kpi-label">Avg Engagement</div><div class="kpi-value">{kpis["avg_engagement"]:,.0f}</div></div>', unsafe_allow_html=True)
with r4:
    st.markdown(f'<div class="kpi-card purple"><div class="kpi-label">Top Category</div><div class="kpi-value">{kpis["top_category"]}</div></div>', unsafe_allow_html=True)

st.markdown("**Category performance table (included in PDF):**")
display_df = cat_df[["category", "article_count", "avg_engagement", "total_views", "total_shares"]].copy()
display_df["avg_engagement"] = display_df["avg_engagement"].apply(lambda x: f"{x:,.0f}")
display_df["total_views"]    = display_df["total_views"].apply(lambda x: f"{x:,}")
display_df["total_shares"]   = display_df["total_shares"].apply(lambda x: f"{x:,}")
display_df = display_df.rename(columns={
    "category": "Category", "article_count": "Articles",
    "avg_engagement": "Avg Engagement", "total_views": "Total Views",
    "total_shares": "Total Shares"
})
st.dataframe(display_df, hide_index=True, use_container_width=True)

st.markdown("**Top 5 articles (included in PDF):**")
preview_top = top_df.copy()
preview_top["title"] = preview_top["title"].str[:60] + "..."
preview_top["engagement_score"] = preview_top["engagement_score"].apply(lambda x: f"{x:,.0f}")
preview_top["views"] = preview_top["views"].apply(lambda x: f"{x:,}")
st.dataframe(
    preview_top[["title", "category", "engagement_score", "views", "author"]],
    hide_index=True,
    use_container_width=True,
    column_config={
        "title": "Headline", "category": "Category",
        "engagement_score": "Engagement", "views": "Views", "author": "Author"
    }
)

st.divider()

# ── Generate button ──
st.markdown("#### Generate & Download")
st.caption("Click below to generate the complete PDF report")

col_btn, col_info = st.columns([1, 3])
with col_btn:
    gen_btn = st.button("📥 Generate PDF Report", type="primary", use_container_width=True)
with col_info:
    st.caption("Report generates in ~3 seconds. Downloads automatically as a PDF file.")

if gen_btn:
    with st.spinner("Generating professional PDF report..."):
        try:
            path = generate_report(df)
            with open(path, "rb") as f:
                pdf_bytes = f.read()
            st.success("✅ Report generated successfully.")
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"newsflow_weekly_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating report: {e}")
