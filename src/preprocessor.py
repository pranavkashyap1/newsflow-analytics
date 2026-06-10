
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

ENGAGEMENT_PROFILE = {
    "Sports":   {"views": (8000, 3000),  "shares": (400, 150),  "comments": (200, 80),  "likes": (600, 200)},
    "Business": {"views": (6000, 2500),  "shares": (250, 100),  "comments": (150, 60),  "likes": (400, 150)},
    "World":    {"views": (5000, 2000),  "shares": (300, 120),  "comments": (180, 70),  "likes": (350, 130)},
    "Sci/Tech": {"views": (7000, 2800),  "shares": (350, 130),  "comments": (160, 65),  "likes": (500, 180)},
}

AUTHORS = [
    "Sarah Mitchell", "James Thornton", "Priya Sharma", "David Chen",
    "Emma Wilson", "Raj Patel", "Lisa Anderson", "Michael Torres",
    "Anjali Gupta", "Chris Evans"
]

def engineer_features(df, sample_size=5000):
    if len(df) > sample_size:
        df = df.groupby("category", group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 4))
        ).reset_index(drop=True)

    df["title"]       = df["title"].astype(str)
    df["description"] = df["description"].astype(str)

    df["title_length"]       = df["title"].str.len()
    df["description_length"] = df["description"].str.len()
    df["word_count"]         = df["description"].str.split().str.len()
    df["title_word_count"]   = df["title"].str.split().str.len()
    df["has_number"]         = df["title"].str.contains(r'\d', regex=True).astype(int)
    df["has_question"]       = df["title"].str.contains(r'\?', regex=True).astype(int)

    n          = len(df)
    categories = df["category"].values

    views    = np.zeros(n, dtype=int)
    shares   = np.zeros(n, dtype=int)
    comments = np.zeros(n, dtype=int)
    likes    = np.zeros(n, dtype=int)

    for cat, profile in ENGAGEMENT_PROFILE.items():
        mask  = categories == cat
        count = mask.sum()
        if count == 0:
            continue
        views[mask]    = np.maximum(100, np.random.normal(profile["views"][0],    profile["views"][1],    count).astype(int))
        shares[mask]   = np.maximum(0,   np.random.normal(profile["shares"][0],   profile["shares"][1],   count).astype(int))
        comments[mask] = np.maximum(0,   np.random.normal(profile["comments"][0], profile["comments"][1], count).astype(int))
        likes[mask]    = np.maximum(0,   np.random.normal(profile["likes"][0],    profile["likes"][1],    count).astype(int))

    df["views"]    = views
    df["shares"]   = shares
    df["comments"] = comments
    df["likes"]    = likes

    df["engagement_score"] = (
        df["views"]    * 0.4 +
        df["shares"]   * 0.3 +
        df["likes"]    * 0.2 +
        df["comments"] * 0.1
    ).round(2)

    # Build dates as Python datetime objects — avoids pandas 2.2/Python 3.13 parsing bug
    base      = datetime.now()
    day_offsets = np.random.randint(0, 365, size=n)
    date_series = pd.Series([base - timedelta(days=int(d)) for d in day_offsets])

    df["published_date"] = date_series.values
    df["month"]          = date_series.dt.month.values
    df["day_of_week"]    = date_series.dt.dayofweek.values
    df["week_number"]    = date_series.dt.isocalendar().week.astype(int).values

    df["author"] = np.random.choice(AUTHORS, size=n)

    return df


def process_and_save(raw_df, output_path="data/processed/articles.csv"):
    import os
    os.makedirs("data/processed", exist_ok=True)
    print("Engineering features...")
    df = engineer_features(raw_df)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} processed articles to {output_path}")
    return df
