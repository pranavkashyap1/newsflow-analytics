
import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, r2_score,
                             accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

MODEL_PATH      = "models/rf_regressor.pkl"
CLASSIFIER_PATH = "models/rf_classifier.pkl"
ENCODER_PATH    = "models/label_encoder.pkl"
TFIDF_PATH      = "models/tfidf_vectorizer.pkl"
TIER_PATH       = "models/tier_boundaries.pkl"

STRUCTURAL_FEATURES = [
    "title_length", "word_count", "title_word_count",
    "has_number", "has_question", "month", "day_of_week",
    "category_encoded"
]

def add_binary_label(df):
    """
    Binary classification: Top Performer vs Average.
    Top 35% by engagement = Top Performer.
    We use 35% threshold to keep classes reasonably balanced.
    """
    threshold = df["engagement_score"].quantile(0.65)
    df["is_top_performer"]  = (df["engagement_score"] >= threshold).astype(int)
    df["performance_label"] = df["is_top_performer"].map({1: "Top Performer", 0: "Average"})
    return df, threshold


def build_feature_matrix(df, tfidf=None, le=None, is_train=True):
    df = df.copy()
    df["title"] = df["title"].astype(str)

    if is_train:
        le = LabelEncoder()
        df["category_encoded"] = le.fit_transform(df["category"])
    else:
        df["category_encoded"] = le.transform(df["category"])

    struct = csr_matrix(df[STRUCTURAL_FEATURES].values.astype(float))

    if is_train:
        tfidf = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        text_features = tfidf.fit_transform(df["title"])
    else:
        text_features = tfidf.transform(df["title"])

    return hstack([struct, text_features]), le, tfidf


def train_models(df):
    os.makedirs("models", exist_ok=True)

    df, threshold = add_binary_label(df)
    dist = df["performance_label"].value_counts()
    print(f"Top performer threshold: engagement >= {threshold:.0f}")
    print(f"Label distribution:\n{dist.to_string()}\n")

    X, le, tfidf = build_feature_matrix(df, is_train=True)
    y_reg = df["engagement_score"]
    y_clf = df["performance_label"]

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf,
        test_size=0.2, random_state=42, stratify=y_clf
    )

    # ── Regression ──
    print("Training engagement score regressor...")
    lr = LinearRegression()
    lr.fit(X_train, yr_train)
    lr_mae = mean_absolute_error(yr_test, lr.predict(X_test))
    lr_r2  = r2_score(yr_test, lr.predict(X_test))

    rf_reg = RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_leaf=4, random_state=42, n_jobs=-1
    )
    rf_reg.fit(X_train, yr_train)
    rf_mae = mean_absolute_error(yr_test, rf_reg.predict(X_test))
    rf_r2  = r2_score(yr_test, rf_reg.predict(X_test))

    print(f"  Baseline (Linear Regression) → MAE: {lr_mae:.2f}, R²: {lr_r2:.4f}")
    print(f"  Random Forest Regressor      → MAE: {rf_mae:.2f}, R²: {rf_r2:.4f}")

    # ── Binary Classifier with class_weight='balanced' ──
    # This tells the model to penalise misclassifying the minority
    # class (Top Performer) more heavily — fixing the recall problem.
    print("\nTraining top performer classifier (balanced)...")
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",   # KEY FIX: penalises minority class misses
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, yc_train)
    clf_preds  = rf_clf.predict(X_test)
    clf_acc    = accuracy_score(yc_test, clf_preds)
    clf_report = classification_report(yc_test, clf_preds)
    clf_cm     = confusion_matrix(yc_test, clf_preds, labels=["Top Performer", "Average"])

    cv_scores = cross_val_score(rf_clf, X, y_clf, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"  Test accuracy:       {clf_acc:.4f}")
    print(f"  Cross-val accuracy:  {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"\n{clf_report}")

    # ── Save everything ──
    with open(MODEL_PATH,      "wb") as f: pickle.dump(rf_reg, f)
    with open(CLASSIFIER_PATH, "wb") as f: pickle.dump(rf_clf, f)
    with open(ENCODER_PATH,    "wb") as f: pickle.dump(le,     f)
    with open(TFIDF_PATH,      "wb") as f: pickle.dump(tfidf,  f)
    with open(TIER_PATH,       "wb") as f: pickle.dump({"threshold": threshold}, f)

    n_struct           = len(STRUCTURAL_FEATURES)
    struct_importances = rf_reg.feature_importances_[:n_struct]

    return {
        "linear_regression":     {"mae": round(lr_mae, 2), "r2": round(lr_r2,  4)},
        "random_forest":         {"mae": round(rf_mae, 2), "r2": round(rf_r2,  4)},
        "classifier_accuracy":   round(clf_acc, 4),
        "cv_mean":               round(cv_scores.mean(), 4),
        "cv_std":                round(cv_scores.std(),  4),
        "classification_report": clf_report,
        "confusion_matrix":      clf_cm,
        "feature_importance":    dict(zip(STRUCTURAL_FEATURES, struct_importances.round(4))),
        "threshold":             threshold,
    }


def load_model():
    with open(MODEL_PATH,      "rb") as f: rf_reg = pickle.load(f)
    with open(CLASSIFIER_PATH, "rb") as f: rf_clf = pickle.load(f)
    with open(ENCODER_PATH,    "rb") as f: le     = pickle.load(f)
    with open(TFIDF_PATH,      "rb") as f: tfidf  = pickle.load(f)
    with open(TIER_PATH,       "rb") as f: tiers  = pickle.load(f)
    return rf_reg, rf_clf, le, tfidf, tiers


def predict_engagement(title, category, rf_reg, rf_clf, le, tfidf):
    title = str(title)
    row = pd.DataFrame([{
        "title":            title,
        "category":         category,
        "title_length":     len(title),
        "word_count":       len(title.split()),
        "title_word_count": len(title.split()),
        "has_number":       int(bool(re.search(r"\d", title))),
        "has_question":     int("?" in title),
        "month":            6,
        "day_of_week":      2,
        "category_encoded": le.transform([category])[0]
    }])

    struct     = csr_matrix(row[STRUCTURAL_FEATURES].values.astype(float))
    text_feats = tfidf.transform(row["title"])
    X          = hstack([struct, text_feats])

    score   = round(float(rf_reg.predict(X)[0]), 2)
    label   = rf_clf.predict(X)[0]
    proba   = rf_clf.predict_proba(X)[0]
    classes = rf_clf.classes_

    return {
        "engagement_score":  score,
        "performance_label": label,
        "probabilities":     dict(zip(classes, (proba * 100).round(1)))
    }
