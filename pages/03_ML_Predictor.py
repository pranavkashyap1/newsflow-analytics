
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml_models import load_model, train_models
from src.nlp_analyzer import load_nlp_model, train_nlp_model, analyze_headline

st.set_page_config(page_title="ML Predictor", page_icon="🤖", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/articles.csv")

@st.cache_resource
def load_all_models():
    try:
        rf_reg, rf_clf, le, tfidf, tiers = load_model()
        nlp = load_nlp_model()
    except FileNotFoundError:
        df = load_data()
        train_models(df)
        train_nlp_model(df)
        rf_reg, rf_clf, le, tfidf, tiers = load_model()
        nlp = load_nlp_model()
    return rf_reg, rf_clf, le, tfidf, tiers, nlp

df = load_data()
rf_reg, rf_clf, le, tfidf, tiers, nlp = load_all_models()

st.title("🤖 ML Predictor & Headline Analyzer")
st.caption("Predict article performance before publishing using Machine Learning")

# ── Headline Analyzer ──
st.subheader("Headline performance analyzer")
st.write("Enter any news headline to get instant performance prediction and editorial insights.")

col1, col2 = st.columns([3, 1])
with col1:
    headline = st.text_input(
        "Headline",
        placeholder="e.g. Apple announces record quarterly revenue beating Wall Street estimates"
    )
with col2:
    st.write("")
    st.write("")
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

if analyze_btn and headline:
    with st.spinner("Running ML pipeline..."):
        result = analyze_headline(headline, nlp, rf_reg, rf_clf, le, tfidf)

    st.divider()

    # ── Prediction result banner ──
    label = result["performance_label"]
    score = result["engagement_score"]
    tp_prob = result["probabilities"].get("Top Performer", 0)

    if label == "Top Performer":
        st.success(f"🚀 **Top Performer** — Predicted engagement score: {score:,.0f} | Confidence: {tp_prob}%")
    else:
        st.warning(f"📊 **Average Performer** — Predicted engagement score: {score:,.0f} | Top performer probability: {tp_prob}%")

    col3, col4, col5 = st.columns(3)
    col3.metric("Predicted category",    result["predicted_category"])
    col4.metric("Engagement score",      f"{score:,.0f}")
    col5.metric("Top performer chance",  f"{tp_prob}%")

    st.divider()
    col6, col7 = st.columns(2)

    with col6:
        st.subheader("Category confidence")
        scores_df = pd.DataFrame(
            list(result["category_scores"].items()),
            columns=["Category", "Confidence (%)"]
        ).sort_values("Confidence (%)", ascending=True)
        fig = px.bar(
            scores_df, x="Confidence (%)", y="Category",
            orientation="h", color="Confidence (%)",
            color_continuous_scale="Teal",
            range_x=[0, 100]
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col7:
        st.subheader("Performance probability")
        prob_df = pd.DataFrame(
            list(result["probabilities"].items()),
            columns=["Label", "Probability (%)"]
        )
        colors = ["#2ecc71" if l == "Top Performer" else "#e74c3c"
                  for l in prob_df["Label"]]
        fig2 = go.Figure(go.Bar(
            x=prob_df["Label"],
            y=prob_df["Probability (%)"],
            marker_color=colors
        ))
        fig2.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="Probability (%)",
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Editorial insights")
    cols = st.columns(2)
    for i, (itype, msg) in enumerate(result["insights"]):
        with cols[i % 2]:
            if itype == "success":
                st.success(msg)
            elif itype == "warning":
                st.warning(msg)
            else:
                st.info(msg)

elif analyze_btn and not headline:
    st.warning("Please enter a headline first.")

st.divider()

# ── Model performance section ──
st.subheader("Model performance summary")

col8, col9, col10 = st.columns(3)
col8.metric("Regressor R²",            "0.1876", delta="vs baseline 0.0058")
col9.metric("Classifier accuracy",     "63.0%",  delta="cross-val: 62.7%")
col10.metric("NLP category accuracy",  "86.9%",  delta="Sports F1: 0.94")

st.divider()
col11, col12 = st.columns(2)

with col11:
    st.subheader("Regression: Random Forest vs Baseline")
    cmp_df = pd.DataFrame({
        "Model":  ["Linear Regression (baseline)", "Random Forest"],
        "MAE":    [925.72, 828.32],
        "R²":     [0.0058, 0.1876]
    })
    st.dataframe(cmp_df, hide_index=True, use_container_width=True)
    st.caption("Random Forest achieves 32× better R² than linear baseline by capturing non-linear feature interactions.")

with col12:
    st.subheader("Feature importance (structural features)")
    fi_df = pd.DataFrame({
        "Feature":    ["category", "description_length", "title_length",
                       "month", "day_of_week", "title_word_count",
                       "word_count", "has_number", "has_question"],
        "Importance": [0.18, 0.16, 0.16, 0.15, 0.11, 0.10, 0.10, 0.03, 0.01]
    }).sort_values("Importance", ascending=True)
    fig3 = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues"
    )
    fig3.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

st.divider()
with st.expander("Why is R² relatively low? (Interview explanation)"):
    st.markdown("""
**This is expected and explainable — not a flaw.**

Engagement prediction is one of the hardest problems in media analytics because:

1. **Simulated data** — our engagement metrics are generated, not real user behaviour
2. **Missing signals** — real models use: author reputation, time of day, trending topics, social amplification, homepage placement
3. **Inherent randomness** — even at top media companies, content virality has a significant random component

**What we did to maximise signal:**
- Combined structural features (title length, word count) with semantic TF-IDF features
- Improved R² by 32× over the linear baseline (0.006 → 0.188)
- Reframed as binary classification to answer the practical editorial question: *"Will this outperform?"*
- Fixed class imbalance with `class_weight='balanced'`, improving Top Performer recall from 6% to 79%

**Real-world next steps:** Add author history, time-of-day features, social trend signals, and word embeddings (BERT).
    """)
