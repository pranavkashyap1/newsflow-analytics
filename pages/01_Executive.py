import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytics import get_kpis, category_performance, top_articles

st.set_page_config(page_title="Executive Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    from datetime import datetime

    df = pd.read_csv("data/processed/articles.csv")

    def safe_parse(x):
        try:
            return datetime.strptime(
                str(x),
                "%Y-%m-%d %H:%M:%S.%f"
            )
        except:
            return None

    df["published_date"] = (
        df["published_date"]
        .apply(safe_parse)
    )

    return df

df = load_data()
kpis = get_kpis(df)

st.title("📊 Executive Dashboard")
st.caption("High-level performance overview across all content categories")

# ── KPI Cards ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Articles",    f"{kpis['total_articles']:,}")
c2.metric("Total Views",       f"{kpis['total_views']:,}")
c3.metric("Avg Engagement",    f"{kpis['avg_engagement']:,.0f}")
c4.metric("Top Category",      kpis['top_category'])

c5, c6, c7, c8 = st.columns(4)
c5.metric("Total Shares",      f"{kpis['total_shares']:,}")
c6.metric("Avg Views/Article", f"{kpis['avg_views']:,}")
c7.metric("Top Author",        kpis['top_author'])
c8.metric("Articles This Week",f"{kpis['articles_this_week']:,}")

st.divider()

# ── Row 1: Category distribution + Engagement by category ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("Article distribution by category")
    cat_df = category_performance(df)
    fig = px.pie(
        cat_df, values="article_count", names="category",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Average engagement score by category")
    fig2 = px.bar(
        cat_df.sort_values("avg_engagement", ascending=True),
        x="avg_engagement", y="category", orientation="h",
        color="avg_engagement",
        color_continuous_scale="Teal",
        labels={"avg_engagement": "Avg engagement score", "category": ""}
    )
    fig2.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Row 2: Total views by category + Top articles table ──
col3, col4 = st.columns(2)

with col3:
    st.subheader("Total views by category")
    fig3 = px.bar(
        cat_df.sort_values("total_views", ascending=False),
        x="category", y="total_views",
        color="category",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"total_views": "Total views", "category": "Category"}
    )
    fig3.update_layout(showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Top 10 articles by engagement")
    top_df = top_articles(df, n=10)
    top_df["title_short"] = top_df["title"].str[:60] + "..."
    st.dataframe(
        top_df[["title_short", "category", "engagement_score", "views"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "title_short":      "Title",
            "category":         "Category",
            "engagement_score": st.column_config.NumberColumn("Engagement", format="%.0f"),
            "views":            st.column_config.NumberColumn("Views", format="%d"),
        }
    )
