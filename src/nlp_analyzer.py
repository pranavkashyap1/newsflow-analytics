
import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

NLP_MODEL_PATH = "models/nlp_pipeline.pkl"

def train_nlp_model(df):
    os.makedirs("models", exist_ok=True)
    df = df.copy()
    df["title"] = df["title"].astype(str)

    X = df["title"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=500,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds    = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report   = classification_report(y_test, preds)

    print(f"NLP Category Classifier Accuracy: {accuracy:.4f}")
    print(report)

    with open(NLP_MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline, accuracy, report


def load_nlp_model():
    with open(NLP_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def analyze_headline(title, nlp_pipeline, rf_reg, rf_clf, le, tfidf):
    title = str(title)

    # Category prediction
    predicted_category = nlp_pipeline.predict([title])[0]
    category_proba     = nlp_pipeline.predict_proba([title])[0]
    categories         = nlp_pipeline.classes_
    category_scores    = dict(zip(categories, (category_proba * 100).round(1)))

    # Engagement prediction
    from src.ml_models import predict_engagement
    prediction = predict_engagement(title, predicted_category, rf_reg, rf_clf, le, tfidf)

    # Editorial insights
    insights = []
    words = title.split()

    if len(words) < 6:
        insights.append(("warning", "Headline is too short — add more context for better SEO and clicks"))
    elif len(words) > 16:
        insights.append(("warning", "Headline is long — consider trimming to under 12 words"))
    else:
        insights.append(("success", f"Headline length is optimal ({len(words)} words)"))

    if re.search(r'\d', title):
        insights.append(("success", "Contains numbers — data-driven headlines get higher CTR"))
    else:
        insights.append(("info", "Adding specific numbers can boost engagement (e.g. '5 reasons', '$2.3B deal')"))

    if "?" in title:
        insights.append(("success", "Question format drives curiosity and click-through"))

    power_words = ["breaking", "exclusive", "urgent", "new", "first", "top",
                   "best", "worst", "record", "major", "crisis", "shock"]
    found = [w for w in power_words if w.lower() in title.lower()]
    if found:
        insights.append(("success", f"Power words detected: {', '.join(found)}"))
    else:
        insights.append(("info", "Consider power words: Breaking, Exclusive, Record, Major"))

    title_lower = title.lower()
    if any(w in title_lower for w in ["says", "announces", "reveals", "confirms"]):
        insights.append(("success", "Action verb present — active voice performs better"))

    return {
        "predicted_category": predicted_category,
        "category_scores":    category_scores,
        "engagement_score":   prediction["engagement_score"],
        "performance_label":  prediction["performance_label"],
        "probabilities":      prediction["probabilities"],
        "insights":           insights,
    }
