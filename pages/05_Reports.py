
import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.report_generator import generate_report
from src.analytics import get_kpis, category_performance

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/articles.csv", quoting=1)

df     = load_data()
kpis   = get_kpis(df)
cat_df = category_performance(df)

st.title("📄 Automated Report Generation")
st.caption("Generate and export professional editorial reports as PDF")

# ── Report preview ──
st.subheader("Report preview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total articles",  f"{kpis['total_articles']:,}")
col2.metric("Total views",     f"{kpis['total_views']:,}")
col3.metric("Avg engagement",  f"{kpis['avg_engagement']:,.0f}")
col4.metric("Top category",    kpis["top_category"])

st.divider()

st.subheader("Category performance preview")
st.dataframe(
    cat_df[["category", "article_count", "avg_engagement", "total_views", "total_shares"]],
    hide_index=True,
    use_container_width=True,
    column_config={
        "category":       "Category",
        "article_count":  "Articles",
        "avg_engagement": st.column_config.NumberColumn("Avg Engagement", format="%.0f"),
        "total_views":    st.column_config.NumberColumn("Total Views",    format="%d"),
        "total_shares":   st.column_config.NumberColumn("Total Shares",   format="%d"),
    }
)

st.divider()

# ── Generate button ──
st.subheader("Generate weekly report")
st.write("Click below to generate a complete PDF report including KPIs, category breakdown, top articles, and editorial recommendations.")

if st.button("Generate PDF report", type="primary"):
    with st.spinner("Generating report..."):
        try:
            path = generate_report(df)
            with open(path, "rb") as f:
                pdf_bytes = f.read()
            st.success("Report generated successfully.")
            st.download_button(
                label="Download PDF report",
                data=pdf_bytes,
                file_name=f"newsflow_weekly_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating report: {e}")
