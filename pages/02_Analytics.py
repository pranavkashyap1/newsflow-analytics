import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytics import category_performance, weekly_trend, author_performance, engagement_distribution

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    from datetime import datetime

    df = pd.read_csv("data/processed/articles.csv", quoting=1)

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

st.title("📈 Analytics Dashboard")
st.caption("Deep-dive into content performance, trends, and engagement patterns")

# ── Sidebar filters ──
st.sidebar.header("Filters")
categories = ["All"] + sorted(df["category"].unique().tolist())
selected_cat = st.sidebar.selectbox("Category", categories)

if selected_cat != "All":
    df = df[df["category"] == selected_cat]

st.sidebar.metric("Filtered articles", f"{len(df):,}")

st.divider()

# ── Row 1: Weekly trend ──
st.subheader("Weekly traffic trend")
trend_df = weekly_trend(df)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=trend_df["week_number"], y=trend_df["total_views"],
    mode="lines+markers", name="Total views",
    line=dict(color="#2ecc71", width=2),
    fill="tozeroy", fillcolor="rgba(46,204,113,0.1)"
))
fig1.add_trace(go.Scatter(
    x=trend_df["week_number"], y=trend_df["avg_engagement"],
    mode="lines+markers", name="Avg engagement",
    line=dict(color="#3498db", width=2),
    yaxis="y2"
))
fig1.update_layout(
    xaxis_title="Week number",
    yaxis_title="Total views",
    yaxis2=dict(title="Avg engagement", overlaying="y", side="right"),
    legend=dict(x=0, y=1),
    margin=dict(t=10, b=10),
    hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── Row 2: Engagement distribution + Shares vs Views ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("Engagement score distribution")
    dist_df = engagement_distribution(df)
    fig2 = px.box(
        dist_df, x="category", y="engagement_score",
        color="category",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"engagement_score": "Engagement score", "category": ""}
    )
    fig2.update_layout(showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Views vs shares by category")
    fig3 = px.scatter(
        df, x="views", y="shares",
        color="category",
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.5, size_max=8,
        labels={"views": "Views", "shares": "Shares"}
    )
    fig3.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Row 3: Author performance + Articles by day of week ──
col3, col4 = st.columns(2)

with col3:
    st.subheader("Author performance")
    auth_df = author_performance(df)
    fig4 = px.bar(
        auth_df, x="avg_engagement", y="author",
        orientation="h", color="avg_engagement",
        color_continuous_scale="Blues",
        labels={"avg_engagement": "Avg engagement", "author": ""}
    )
    fig4.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    st.subheader("Publishing pattern by day of week")
    day_map = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    day_df = df.groupby("day_of_week")["engagement_score"].mean().reset_index()
    day_df["day"] = day_df["day_of_week"].map(day_map)
    fig5 = px.bar(
        day_df, x="day", y="engagement_score",
        color="engagement_score", color_continuous_scale="Oranges",
        labels={"engagement_score": "Avg engagement", "day": "Day"}
    )
    fig5.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig5, use_container_width=True)
