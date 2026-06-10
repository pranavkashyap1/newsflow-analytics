import pandas as pd
import os

CATEGORY_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

def load_raw_data(path="data/raw/train.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={"text": "title", "label": "class_index"})

    df["class_index"] = pd.to_numeric(df["class_index"], errors="coerce")
    df = df.dropna(subset=["class_index"])
    df["class_index"] = df["class_index"].astype(int)

    df["category"]    = df["class_index"].map(CATEGORY_MAP)
    df["description"] = df["title"]

    df = df.dropna(subset=["title", "category"])
    df = df.drop(columns=["class_index"])
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df):,} articles across {df['category'].nunique()} categories")
    return df
