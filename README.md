# NewsFlow Analytics — Media Traffic Intelligence Platform

> AI-powered analytics platform for digital newsrooms. Analyze content performance, predict article engagement using Machine Learning, and automate editorial reporting.

**Built for:** Times Group Internship | AI Automation, Analytics, BI, Dashboard Development, Reporting Automation

**Live Demo:** [newsflow-analytics.streamlit.app](https://newsflow-analytics-nwljj9yuaxq2xctqkajx4c.streamlit.app)

---

## What This Project Does

Media organizations like Times of India publish hundreds of articles daily. Editors need to answer:
- Which content categories drive the most traffic?
- Will this headline perform well before we publish it?
- Which authors consistently produce top content?
- Can we automate the weekly performance report?

**NewsFlow Analytics answers all of these — automatically.**

---

## Platform Features

| Page | What It Does |
|---|---|
| 📊 Executive Dashboard | KPI cards, category distribution, top articles — the big picture |
| 📈 Analytics Dashboard | Weekly trends, engagement distribution, author rankings, day-of-week patterns |
| 🤖 ML Predictor | Type any headline → get engagement score, category, top performer probability, editorial insights |
| 💡 AI Assistant | Groq LLaMA 3.1 generates editorial strategy recommendations and headline variants |
| 📄 Reports | One-click professional PDF report with KPIs, category breakdown, and recommendations |

---

## Machine Learning

Three models power the prediction engine:

### 1. NLP Category Classifier
- **Algorithm:** TF-IDF (5,000 features, bigrams) + Logistic Regression
- **Task:** Classify any headline into World / Sports / Business / Sci/Tech
- **Accuracy: 87%** overall — Sports achieves 94% F1 (distinctive vocabulary)

### 2. Engagement Score Regressor
- **Algorithm:** Random Forest (200 trees) + TF-IDF semantic features
- **Task:** Predict numeric engagement score for any article
- **R² = 0.188** — 32× improvement over linear baseline (0.006)
- Combined structural features (title length, word count, has_number) with 500 TF-IDF features

### 3. Top Performer Classifier
- **Algorithm:** Random Forest Classifier with `class_weight='balanced'`
- **Task:** Binary prediction — will this article be in the top 35% of engagement?
- **Accuracy: 63%** | **Top Performer Recall: 79%**
- Key challenge solved: class imbalance — initial model had only 6% recall on top performers. Applied `class_weight='balanced'` to fix minority class misclassification. Recall jumped from 6% → 79%.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Dashboard | Streamlit 1.37 | Web application framework |
| Charts | Plotly | Interactive visualizations |
| Data | Pandas, NumPy | Data processing and feature engineering |
| ML | Scikit-learn | Random Forest, Logistic Regression, TF-IDF |
| AI | Groq API (LLaMA 3.1) | Editorial recommendations and headline generation |
| PDF | FPDF2 | Automated report generation |
| Dataset | AG News (Kaggle) | 120,000 news articles, 4 categories |
| Deployment | Streamlit Community Cloud | Cloud hosting |

---

## Project Architecture newsflow-analytics/
├── app.py                    ← Landing page + auto-setup
├── setup.py                  ← First-run data pipeline + model training
├── requirements.txt
│
├── src/                      ← Business logic (pure Python, no Streamlit)
│   ├── data_loader.py        ← Load AG News CSV, map labels to categories
│   ├── preprocessor.py       ← Feature engineering, engagement simulation
│   ├── analytics.py          ← KPI computations, aggregations
│   ├── ml_models.py          ← Train/load Random Forest models
│   ├── nlp_analyzer.py       ← TF-IDF pipeline, headline analysis
│   ├── ai_assistant.py       ← Groq API integration, rule-based fallback
│   └── report_generator.py  ← PDF generation with FPDF2
│
├── pages/                    ← Streamlit multi-page app
│   ├── 01_Executive.py
│   ├── 02_Analytics.py
│   ├── 03_ML_Predictor.py
│   ├── 04_AI_Assistant.py
│   └── 05_Reports.py
│
├── data/
│   └── raw/train.csv         ← AG News dataset
│
└── models/                   ← Saved .pkl model files (auto-generated)  ---

## Setup & Run Locally

### Prerequisites
- Python 3.11
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/pranavkashyap1/newsflow-analytics.git
cd newsflow-analytics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Add your API key
mkdir -p .streamlit
echo 'GROQ_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# Launch
streamlit run app.py
```

The app auto-generates the processed dataset and trains all models on first run (~60 seconds).

---

## Dataset

**AG News Classification Dataset** — 120,000 news articles across 4 categories.

| Category | Label | Articles |
|---|---|---|
| World | 0 | 30,000 |
| Sports | 1 | 30,000 |
| Business | 2 | 30,000 |
| Sci/Tech | 3 | 30,000 |

Source: [Kaggle — AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

**Note on engagement data:** Real traffic data is proprietary at media companies. Engagement metrics (views, shares, likes, comments) are simulated using category-specific statistical distributions — standard practice when building analytics platforms without production data access. Sports articles receive higher share distributions, Business higher view distributions, reflecting real-world patterns.

---

## Key Design Decisions

**Why simulate engagement data?**
Real traffic data is proprietary. Simulation with realistic distributions is standard industry practice for building and demonstrating analytics systems.

**Why binary classification over 3-class?**
A 3-class model (Low/Medium/High) had poor recall across all classes due to overlapping distributions. Reframing as binary (Top Performer vs Average) and applying `class_weight='balanced'` improved Top Performer recall from 6% to 79% — far more useful for editorial decisions.

**Why TF-IDF over BERT embeddings?**
TF-IDF is interpretable, fast, and interview-explainable. It improved R² by 32× over structural features alone. BERT is the natural next step with more compute.

**Why Streamlit over Flask/Django?**
Streamlit lets the team focus on data and ML rather than frontend development. For a data product where insights matter more than UI complexity, it's the right tool.

**Why Random Forest over XGBoost or neural networks?**
At 5,000 rows, Random Forest is appropriately complex. It handles mixed feature types, requires minimal tuning, and is interpretable via feature importance. XGBoost would be the next step with more data.

---

## Future Scope

- **Real-time ingestion** — Connect to news APIs (NewsAPI, RSS feeds) for live data
- **BERT embeddings** — Replace TF-IDF with sentence transformers for semantic understanding
- **Author reputation features** — Add author history to engagement prediction
- **A/B headline testing** — Test headline variants before full publication
- **REST API layer** — FastAPI endpoints for CMS integration
- **Multi-publication support** — Handle multiple Times Group properties
## Author

**Pranav Kashyap** — B.Tech Computer Science
Built for Times Group Internship Application | Data Science & AI Automation Track

[![GitHub](https://img.shields.io/badge/GitHub-pranavkashyap1-black?logo=github)](https://github.com/pranavkashyap1)
