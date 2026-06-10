import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml_models import load_model, train_models
from src.nlp_analyzer import load_nlp_model, train_nlp_model, analyze_headline

st.set_page_config(page_title="ML Predictor", page_icon="🤖", layout="wide")

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
.kpi-card.green  { border-left-color: #4CAF50; }
.kpi-card.orange { border-left-color: #FF9800; }
.kpi-card.purple { border-left-color: #9C27B0; }
.kpi-label {
    font-size: 12px;
    color: #aaa;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
}
.result-card {
    background: linear-gradient(135deg, #0d2137, #0a2744);
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #1a4a7a;
    margin: 16px 0;
}
.result-card.top {
    background: linear-gradient(135deg, #0d3320, #0a3d1f);
    border: 1px solid #1a7a3a;
}
.result-title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
}
.result-sub {
    font-size: 14px;
    color: #aaa;
}
.insight-box {
    background: #1a1d2e;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    color: #ccc;
    margin: 6px 0;
    border-left: 3px solid #2196F3;
}
.insight-box.success { border-left-color: #4CAF50; }
.insight-box.warning { border-left-color: #FF9800; }
.insight-box.info    { border-left-color: #2196F3; }
.sample-btn {
    background: #1e2130;
    border: 1px solid #2a2d3e;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    color: #aaa;
    cursor: pointer;
    margin: 4px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/articles.csv", quoting=1)

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

# ── Header ──
st.markdown("## 🤖 ML Predictor & Headline Analyzer")
st.caption("Type any news headline below — our ML models will predict its engagement score, category, and whether it will be a top performer")
st.divider()

# ── Sample headlines ──
st.markdown("#### Try a sample headline or write your own")
samples = [
    "Apple announces record quarterly revenue beating Wall Street estimates",
    "India wins the Cricket World Cup in dramatic final against Australia",
    "Scientists discover new treatment for Alzheimer's disease",
    "Federal Reserve raises interest rates by 50 basis points amid inflation",
    "Tesla unveils fully autonomous self-driving software update",
]

selected_sample = st.selectbox(
    "Pick a sample headline to analyze:",
    ["— select a sample or type below —"] + samples
)

headline = st.text_input(
    "Or type your own headline:",
    value=selected_sample if selected_sample != "— select a sample or type below —" else "",
    placeholder="e.g. RBI cuts interest rates for the first time in 5 years"
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze_btn and headline and headline != "— select a sample or type below —":
    with st.spinner("Running ML pipeline — category classification → engagement prediction → performance scoring..."):
        result = analyze_headline(headline, nlp, rf_reg, rf_clf, le, tfidf)

    st.divider()

    # ── Result banner ──
    label    = result["performance_label"]
    score    = result["engagement_score"]
    tp_prob  = result["probabilities"].get("Top Performer", 0)
    category = result["predicted_category"]

    if label == "Top Performer":
        st.markdown(f"""
        <div class="result-card top">
            <div class="result-title">🚀 Top Performer Predicted</div>
            <div class="result-sub">This headline is likely to outperform 65% of published articles based on our ML model</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">📊 Average Performer Predicted</div>
            <div class="result-sub">Consider the editorial insights below to improve this headline's performance potential</div>
        </div>""", unsafe_allow_html=True)

    # ── Metrics row ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="kpi-card blue"><div class="kpi-label">Predicted Category</div><div class="kpi-value" style="font-size:20px">{category}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="kpi-card orange"><div class="kpi-label">Engagement Score</div><div class="kpi-value">{score:,.0f}</div></div>', unsafe_allow_html=True)
    with m3:
        color = "green" if tp_prob >= 50 else "purple"
        st.markdown(f'<div class="kpi-card {color}"><div class="kpi-label">Top Performer Chance</div><div class="kpi-value">{tp_prob}%</div></div>', unsafe_allow_html=True)
    with m4:
        word_count = len(headline.split())
        wc_color = "green" if 6 <= word_count <= 14 else "orange"
        st.markdown(f'<div class="kpi-card {wc_color}"><div class="kpi-label">Headline Word Count</div><div class="kpi-value">{word_count} words</div></div>', unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Category Confidence")
        st.caption("How confident is the model about the category prediction?")
        scores_df = pd.DataFrame(
            list(result["category_scores"].items()),
            columns=["Category", "Confidence (%)"]
        ).sort_values("Confidence (%)", ascending=True)

        fig = px.bar(
            scores_df, x="Confidence (%)", y="Category",
            orientation="h",
            color="Confidence (%)",
            color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#2196F3"], [1, "#64B5F6"]],
            range_x=[0, 100],
            text="Confidence (%)"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff",
            xaxis=dict(gridcolor="#2a2d3e"),
            margin=dict(t=10, b=10, r=60)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Performance Probability")
        st.caption("Probability of being a Top Performer vs Average")

        prob_df = pd.DataFrame(
            list(result["probabilities"].items()),
            columns=["Label", "Probability (%)"]
        )
        bar_colors = ["#4CAF50" if l == "Top Performer" else "#f44336" for l in prob_df["Label"]]

        fig2 = go.Figure(go.Bar(
            x=prob_df["Label"],
            y=prob_df["Probability (%)"],
            marker_color=bar_colors,
            text=prob_df["Probability (%)"],
            texttemplate="%{text:.1f}%",
            textposition="outside"
        ))
        fig2.update_layout(
            yaxis_range=[0, 110],
            yaxis_title="Probability (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff",
            yaxis=dict(gridcolor="#2a2d3e"),
            margin=dict(t=20, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Editorial insights ──
    st.markdown("#### Editorial Insights")
    st.caption("Specific recommendations to improve this headline's performance")

    cols = st.columns(2)
    for i, (itype, msg) in enumerate(result["insights"]):
        with cols[i % 2]:
            icon = "✅" if itype == "success" else ("⚠️" if itype == "warning" else "💡")
            css_class = "success" if itype == "success" else ("warning" if itype == "warning" else "info")
            st.markdown(f'<div class="insight-box {css_class}">{icon} {msg}</div>', unsafe_allow_html=True)

elif analyze_btn and not headline:
    st.warning("Please enter or select a headline first.")

st.divider()

# ── Model performance section ──
st.markdown("#### How Our Models Were Built")
st.caption("Transparency about model performance — important for production use")

tab1, tab2, tab3 = st.tabs(["📊 Engagement Regressor", "🎯 Top Performer Classifier", "🏷️ Category Classifier"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What it does:** Predicts a numeric engagement score for any article")
        st.markdown("**Algorithm:** Random Forest (200 trees)")
        st.markdown("**Features:** 8 structural features + 500 TF-IDF semantic features")
        perf_df = pd.DataFrame({
            "Model":  ["Linear Regression (baseline)", "Random Forest"],
            "MAE":    [925.72, 828.32],
            "R²":     [0.0058, 0.1876],
            "Improvement": ["—", "32× better R²"]
        })
        st.dataframe(perf_df, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Why R² is 0.19 (not higher):**")
        st.info("Engagement prediction is inherently noisy. Even Netflix can't perfectly predict virality. We're missing key signals: author reputation, trending topics, publish time, homepage placement. Our value is the directional signal — 32× improvement over random guessing.")

with tab2:
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**What it does:** Predicts if an article will be a Top Performer (top 35%)")
        st.markdown("**Algorithm:** Random Forest Classifier with `class_weight='balanced'`")
        clf_df = pd.DataFrame({
            "Metric":   ["Accuracy", "Top Performer Recall", "Macro F1", "Cross-val Accuracy"],
            "Before fix": ["66%", "6%", "0.45", "—"],
            "After fix":  ["63%", "79%", "0.63", "62.7%"]
        })
        st.dataframe(clf_df, hide_index=True, use_container_width=True)
    with c4:
        st.markdown("**The class imbalance problem we solved:**")
        st.info("Initial model had 79% recall on Average but only 6% on Top Performers — it was predicting Average for everything. Applied class_weight='balanced' to penalize minority class misses. Top Performer recall jumped from 6% to 79%. This is more useful for editorial decisions.")

with tab3:
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**What it does:** Classifies any headline into World / Sports / Business / Sci-Tech")
        st.markdown("**Algorithm:** TF-IDF (5000 features, bigrams) + Logistic Regression")
        nlp_df = pd.DataFrame({
            "Category": ["World", "Sports", "Business", "Sci/Tech", "Overall"],
            "Precision": ["0.87", "0.94", "0.84", "0.83", "0.87"],
            "Recall":    ["0.88", "0.95", "0.81", "0.84", "0.87"],
            "F1 Score":  ["0.87", "0.94", "0.83", "0.83", "0.87"]
        })
        st.dataframe(nlp_df, hide_index=True, use_container_width=True)
    with c6:
        st.markdown("**Why Sports has the highest accuracy (94% F1):**")
        st.info("Sports vocabulary is highly distinctive — words like 'wicket', 'touchdown', 'Champions League' only appear in Sports articles. TF-IDF gives these words high weights, making Sports the easiest category to classify.")
