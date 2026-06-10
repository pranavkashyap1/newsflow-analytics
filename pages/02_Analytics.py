import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytics import category_performance, weekly_trend, author_performance, engagement_distribution

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

st.markdown("""
<style>
.insight-box {
    background: #1a1d2e;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    color: #aaa;
    margin-top: 8px;
    border-left: 3px solid #4CAF50;
}
.filter-label {
    font-size: 13px;
    color: #aaa;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

COLORS = {
    "Sports":   "#2196F3",
    "Business": "#4CAF50",
    "World":    "#FF9800",
    "Sci/Tech": "#9C27B0"
}

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/articles.csv", quoting=1)

df = load_data()

# ── Header ──
st.markdown("## 📈 Analytics Dashboard")
st.caption("Deep-dive into content performance, trends, and engagement patterns across categories and time")
st.divider()

# ── Sidebar filters ──
st.sidebar.markdown("### Filters")
categories   = ["All"] + sorted(df["category"].unique().tolist())
selected_cat = st.sidebar.selectbox("Category", categories)
st.sidebar.markdown("---")

if selected_cat != "All":
    filtered = df[df["category"] == selected_cat]
else:
    filtered = df

st.sidebar.metric("Articles shown", f"{len(filtered):,}")
st.sidebar.metric("Avg engagement", f"{filtered['engagement_score'].mean():,.0f}")
st.sidebar.metric("Total views",    f"{filtered['views'].sum()/1_000_000:.1f}M")

# ── Weekly trend ──
st.markdown("#### Weekly Traffic & Engagement Trend")
st.caption("How views and engagement scores have moved over the past 52 weeks")

trend_df = weekly_trend(filtered)
# 4-week moving average
trend_df["views_ma"] = trend_df["total_views"].rolling(4, min_periods=1).mean()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=trend_df["week_number"], y=trend_df["total_views"],
    mode="lines", name="Weekly views",
    line=dict(color="#2196F3", width=1.5),
    opacity=0.4,
    fill="tozeroy", fillcolor="rgba(33,150,243,0.05)"
))
fig1.add_trace(go.Scatter(
    x=trend_df["week_number"], y=trend_df["views_ma"],
    mode="lines", name="4-week avg views",
    line=dict(color="#2196F3", width=2.5),
))
fig1.add_trace(go.Scatter(
    x=trend_df["week_number"], y=trend_df["avg_engagement"],
    mode="lines", name="Avg engagement",
    line=dict(color="#4CAF50", width=2),
    yaxis="y2"
))
fig1.update_layout(
    xaxis_title="Week Number",
    yaxis_title="Total Views",
    yaxis2=dict(title="Avg Engagement Score", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#ffffff",
    xaxis=dict(gridcolor="#2a2d3e"),
    yaxis=dict(gridcolor="#2a2d3e"),
    margin=dict(t=40, b=20),
    hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)
peak_week = trend_df.loc[trend_df["total_views"].idxmax(), "week_number"]
st.markdown(f'<div class="insight-box">💡 Peak traffic occurred in week <b>{peak_week}</b>. The blue shaded area shows raw weekly views — the solid line is the 4-week moving average which smooths out noise.</div>', unsafe_allow_html=True)

st.divider()

# ── Row 2: Engagement distribution + Views vs Shares ──
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Engagement Score Distribution by Category")
    st.caption("How spread out are engagement scores within each category?")

    cat_colors = [COLORS.get(c, "#888") for c in filtered["category"].unique()]
    fig2 = px.violin(
        filtered, x="category", y="engagement_score",
        color="category",
        color_discrete_map=COLORS,
        box=True,
        labels={"engagement_score": "Engagement Score", "category": ""},
    )
    fig2.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        yaxis=dict(gridcolor="#2a2d3e"),
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div class="insight-box">💡 The wider the violin, the more articles cluster at that score. A narrow top means few articles reach peak engagement.</div>', unsafe_allow_html=True)

with col2:
    st.markdown("#### Views vs Shares — Scatter Analysis")
    st.caption("Do high-view articles also get shared? Look for the correlation.")

    fig3 = px.scatter(
        filtered, x="views", y="shares",
        color="category",
        color_discrete_map=COLORS,
        opacity=0.6,
        
        labels={"views": "Views", "shares": "Shares"},
        hover_data=["title", "author"]
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        xaxis=dict(gridcolor="#2a2d3e"),
        yaxis=dict(gridcolor="#2a2d3e"),
        margin=dict(t=20, b=20),
        legend=dict(title="Category")
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="insight-box">💡 Each dot is one article. The trend line shows the relationship between views and shares. Dots above the line = unusually high shareability.</div>', unsafe_allow_html=True)

st.divider()

# ── Row 3: Author performance + Day of week ──
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Author Performance Ranking")
    st.caption("Average engagement score per author — who consistently produces top content?")

    auth_df = author_performance(filtered).head(10)
    auth_df["rank"] = range(1, len(auth_df) + 1)

    fig4 = px.bar(
        auth_df, x="avg_engagement", y="author",
        orientation="h",
        color="avg_engagement",
        color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#2196F3"], [1, "#64B5F6"]],
        text="avg_engagement",
        labels={"avg_engagement": "Avg Engagement", "author": ""}
    )
    fig4.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig4.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        xaxis=dict(gridcolor="#2a2d3e"),
        margin=dict(t=20, b=20, r=60)
    )
    st.plotly_chart(fig4, use_container_width=True)
    top_auth = auth_df.iloc[0]["author"]
    top_eng  = auth_df.iloc[0]["avg_engagement"]
    st.markdown(f'<div class="insight-box">💡 <b>{top_auth}</b> leads with avg engagement of <b>{top_eng:,.0f}</b>. Assign them to high-priority stories for maximum impact.</div>', unsafe_allow_html=True)

with col4:
    st.markdown("#### Best Publishing Days")
    st.caption("Which day of the week produces the highest average engagement?")

    day_map   = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_df    = filtered.groupby("day_of_week")["engagement_score"].mean().reset_index()
    day_df["day"] = day_df["day_of_week"].map(day_map)
    day_df    = day_df.set_index("day").reindex(day_order).reset_index()

    best_day  = day_df.loc[day_df["engagement_score"].idxmax(), "day"]
    worst_day = day_df.loc[day_df["engagement_score"].idxmin(), "day"]

    bar_colors = ["#2196F3" if d == best_day else "#1e3a5f" for d in day_df["day"]]

    fig5 = go.Figure(go.Bar(
        x=day_df["day"],
        y=day_df["engagement_score"],
        marker_color=bar_colors,
        text=day_df["engagement_score"].round(0),
        texttemplate="%{text:,.0f}",
        textposition="outside"
    ))
    fig5.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Avg Engagement Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        yaxis=dict(gridcolor="#2a2d3e"),
        margin=dict(t=30, b=20, r=40)
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(f'<div class="insight-box">💡 <b>{best_day}</b> is the best publishing day — highest average engagement. Avoid <b>{worst_day}</b> for important articles.</div>', unsafe_allow_html=True)

st.divider()

# ── Category comparison table ──
st.markdown("#### Category Performance Summary Table")
st.caption("All metrics side by side — use this for editorial planning meetings")

display_df = category_performance(filtered).copy()
display_df["avg_engagement"]  = display_df["avg_engagement"].apply(lambda x: f"{x:,.0f}")
display_df["total_views"]     = display_df["total_views"].apply(lambda x: f"{x:,}")
display_df["avg_views"]       = display_df["avg_views"].apply(lambda x: f"{x:,.0f}")
display_df["total_shares"]    = display_df["total_shares"].apply(lambda x: f"{x:,}")
display_df["total_likes"]     = display_df["total_likes"].apply(lambda x: f"{x:,}")
display_df["total_comments"]  = display_df["total_comments"].apply(lambda x: f"{x:,}")
display_df = display_df.rename(columns={
    "category":       "Category",
    "article_count":  "Articles",
    "avg_engagement": "Avg Engagement",
    "total_views":    "Total Views",
    "avg_views":      "Avg Views",
    "total_shares":   "Total Shares",
    "total_likes":    "Total Likes",
    "total_comments": "Total Comments",
})
st.dataframe(display_df, hide_index=True, use_container_width=True)
