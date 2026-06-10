import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytics import get_kpis, category_performance, top_articles

st.set_page_config(page_title="Executive Dashboard", page_icon="📊", layout="wide")

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
.kpi-card.orange { border-left-color: #FF9800; }
.kpi-card.purple { border-left-color: #9C27B0; }
.kpi-card.green  { border-left-color: #4CAF50; }
.kpi-card.red    { border-left-color: #f44336; }
.kpi-card.teal   { border-left-color: #00BCD4; }
.kpi-label { font-size: 12px; color: #aaa; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.8px; }
.kpi-value { font-size: 26px; font-weight: 700; color: #ffffff; }
.insight-box {
    background: #1a1d2e;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    color: #aaa;
    margin-top: 8px;
    border-left: 3px solid #2196F3;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/articles.csv", quoting=1)
    return df

df   = load_data()
kpis = get_kpis(df)
cat_df = category_performance(df)

st.title("📊 Executive Dashboard")
st.caption("High-level performance overview across all content categories")

# ── Row 1 KPIs ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Total Articles</div><div class="kpi-value">{kpis["total_articles"]:,}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Total Views</div><div class="kpi-value">{kpis["total_views"]/1_000_000:.1f}M</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi-card orange"><div class="kpi-label">Avg Engagement</div><div class="kpi-value">{kpis["avg_engagement"]:,.0f}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi-card purple"><div class="kpi-label">Top Category</div><div class="kpi-value">{kpis["top_category"]}</div></div>', unsafe_allow_html=True)

# ── Row 2 KPIs ──
c5, c6, c7, c8 = st.columns(4)
with c5:
    st.markdown(f'<div class="kpi-card teal"><div class="kpi-label">Total Shares</div><div class="kpi-value">{kpis["total_shares"]/1_000:.0f}K</div></div>', unsafe_allow_html=True)
with c6:
    st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Avg Views / Article</div><div class="kpi-value">{kpis["avg_views"]:,}</div></div>', unsafe_allow_html=True)
with c7:
    st.markdown(f'<div class="kpi-card green"><div class="kpi-label">Top Author</div><div class="kpi-value" style="font-size:18px">{kpis["top_author"]}</div></div>', unsafe_allow_html=True)
with c8:
    st.markdown(f'<div class="kpi-card orange"><div class="kpi-label">Articles This Week</div><div class="kpi-value">{kpis["articles_this_week"]:,}</div></div>', unsafe_allow_html=True)

st.divider()

# ── Row 1 Charts ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("Category distribution")
    fig = px.pie(
        cat_df, values="article_count", names="category",
        color_discrete_sequence=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
        hole=0.45
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        showlegend=False, margin=dict(t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight-box">💡 All four categories have equal article distribution — 25% each. This is by design to ensure unbiased model training.</div>', unsafe_allow_html=True)

with col2:
    st.subheader("Average engagement by category")
    sorted_cat = cat_df.sort_values("avg_engagement", ascending=True)
    colors = ["#2196F3" if c == kpis["top_category"] else "#374151" for c in sorted_cat["category"]]
    fig2 = go.Figure(go.Bar(
        x=sorted_cat["avg_engagement"],
        y=sorted_cat["category"],
        orientation="h",
        marker_color=colors,
        text=sorted_cat["avg_engagement"].apply(lambda x: f"{x:,.0f}"),
        textposition="outside"
    ))
    fig2.update_layout(
        margin=dict(t=10, b=10, r=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#666"),
        yaxis=dict(color="#ccc"),
        font=dict(color="#ccc")
    )
    st.plotly_chart(fig2, use_container_width=True)
    top = cat_df.loc[cat_df["avg_engagement"].idxmax()]
    bottom = cat_df.loc[cat_df["avg_engagement"].idxmin()]
    gap = top["avg_engagement"] - bottom["avg_engagement"]
    st.markdown(f'<div class="insight-box">💡 <b>{top["category"]}</b> leads with {top["avg_engagement"]:,.0f} avg engagement — {gap:.0f} points ahead of <b>{bottom["category"]}</b>.</div>', unsafe_allow_html=True)

st.divider()

# ── Row 2 Charts ──
col3, col4 = st.columns(2)

with col3:
    st.subheader("Total views by category")
    sorted_views = cat_df.sort_values("total_views", ascending=False)
    fig3 = px.bar(
        sorted_views, x="category", y="total_views",
        color="category",
        color_discrete_sequence=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
        labels={"total_views": "Total Views", "category": ""},
        text=sorted_views["total_views"].apply(lambda x: f"{x/1_000_000:.1f}M")
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(
        showlegend=False, margin=dict(t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#ccc"),
        yaxis=dict(showgrid=False, color="#666"),
        font=dict(color="#ccc")
    )
    st.plotly_chart(fig3, use_container_width=True)
    top_views = cat_df.loc[cat_df["total_views"].idxmax()]
    st.markdown(f'<div class="insight-box">💡 <b>{top_views["category"]}</b> generates the most total views — {top_views["total_views"]/1_000_000:.1f}M across {top_views["article_count"]} articles.</div>', unsafe_allow_html=True)

with col4:
    st.subheader("Top 10 articles by engagement")
    top_df = top_articles(df, n=10)
    top_df["title_short"] = top_df["title"].str[:55] + "..."
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
    st.markdown('<div class="insight-box">💡 Top articles are ranked by weighted engagement score combining views (40%), shares (30%), likes (20%), and comments (10%).</div>', unsafe_allow_html=True)
