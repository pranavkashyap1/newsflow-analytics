import pandas as pd
import numpy as np

def get_kpis(df):
    return {
        "total_articles":     len(df),
        "total_views":        int(df["views"].sum()),
        "avg_engagement":     round(df["engagement_score"].mean(), 2),
        "top_category":       df.groupby("category")["engagement_score"].mean().idxmax(),
        "total_shares":       int(df["shares"].sum()),
        "avg_views":          int(df["views"].mean()),
        "top_author":         df.groupby("author")["engagement_score"].mean().idxmax(),
        "articles_this_week": int((df["week_number"] == df["week_number"].max()).sum()),
    }

def category_performance(df):
    return df.groupby("category").agg(
        article_count    = ("title", "count"),
        avg_engagement   = ("engagement_score", "mean"),
        total_views      = ("views", "sum"),
        avg_views        = ("views", "mean"),
        total_shares     = ("shares", "sum"),
        avg_shares       = ("shares", "mean"),
        total_likes      = ("likes", "sum"),
        total_comments   = ("comments", "sum"),
    ).round(2).reset_index()

def weekly_trend(df):
    return df.groupby("week_number").agg(
        total_views    = ("views", "sum"),
        avg_engagement = ("engagement_score", "mean"),
        article_count  = ("title", "count"),
    ).reset_index().sort_values("week_number")

def top_articles(df, n=10):
    safe_cols = ["title", "category", "engagement_score", "views", "shares", "author"]
    return df.nlargest(n, "engagement_score")[safe_cols].reset_index(drop=True)

def author_performance(df):
    return df.groupby("author").agg(
        article_count  = ("title", "count"),
        avg_engagement = ("engagement_score", "mean"),
        total_views    = ("views", "sum"),
    ).round(2).reset_index().sort_values("avg_engagement", ascending=False)

def engagement_distribution(df):
    return df[["category", "engagement_score", "views", "shares"]]
