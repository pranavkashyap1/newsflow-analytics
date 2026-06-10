import os

def _get_api_key():
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")

def get_model():
    try:
        from groq import Groq
        api_key = _get_api_key()
        if not api_key or api_key == "your_groq_api_key_here":
            return None
        return Groq(api_key=api_key)
    except Exception:
        return None

def _groq_complete(client, prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content

def generate_editorial_recommendations(kpis, category_df, top_articles_df):
    client = get_model()
    if client is None:
        return _rule_based_recommendations(kpis, category_df)

    category_summary = category_df[
        ["category", "avg_engagement", "total_views", "article_count"]
    ].to_string(index=False)

    top_titles = "\n".join(
        f"- [{row['category']}] {row['title'][:80]} (engagement: {row['engagement_score']:.0f})"
        for _, row in top_articles_df.head(5).iterrows()
    )

    prompt = f"""
You are a senior editorial strategist at a major digital media company.
Analyze the following content performance data and provide actionable recommendations.

PLATFORM METRICS:
- Total articles: {kpis['total_articles']:,}
- Total views: {kpis['total_views']:,}
- Average engagement score: {kpis['avg_engagement']:,.0f}
- Top performing category: {kpis['top_category']}
- Top author: {kpis['top_author']}

CATEGORY PERFORMANCE:
{category_summary}

TOP 5 ARTICLES THIS PERIOD:
{top_titles}

Provide exactly the following:
1. EXECUTIVE SUMMARY (2 sentences max)
2. TOP 3 OPPORTUNITIES (specific, actionable)
3. UNDERPERFORMING AREAS (what to fix and how)
4. CONTENT STRATEGY FOR NEXT WEEK (1 specific recommendation)
5. PREDICTED GROWTH LEVER (one insight that could move the needle most)

Be specific, data-driven, and concise. Think like a media executive.
"""
    try:
        return {"source": "groq", "content": _groq_complete(client, prompt)}
    except Exception:
        return _rule_based_recommendations(kpis, category_df)


def generate_headline_suggestions(category, topic, performance_data):
    client = get_model()
    if client is None:
        return _rule_based_headlines(category, topic)

    prompt = f"""
You are an expert digital news editor specializing in high-engagement headlines.

Generate 5 optimized headlines for:
- Category: {category}
- Topic: {topic}
- Target: maximize engagement and click-through rate

Rules:
- Each headline must be under 15 words
- Mix question, number, and statement formats
- Use power words where natural
- Be factually plausible for a news context

Return ONLY the 5 headlines as a numbered list. No explanations.
"""
    try:
        text = _groq_complete(client, prompt)
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        headlines = [l.lstrip("0123456789.-) ") for l in lines if l and l[0].isdigit()]
        return {"source": "groq", "headlines": headlines[:5]}
    except Exception:
        return _rule_based_headlines(category, topic)


def _rule_based_recommendations(kpis, category_df):
    top_cat     = kpis["top_category"]
    total_views = kpis["total_views"]
    avg_eng     = kpis["avg_engagement"]
    sorted_df   = category_df.sort_values("avg_engagement", ascending=False)
    bottom_cat  = sorted_df.iloc[-1]["category"]
    top_eng     = sorted_df.iloc[0]["avg_engagement"]
    bot_eng     = sorted_df.iloc[-1]["avg_engagement"]
    gap         = top_eng - bot_eng

    content = f"""
## Executive Summary
{top_cat} content is the strongest performer with the highest average engagement score on the platform.
Total platform views stand at {total_views:,} with an average engagement score of {avg_eng:,.0f}.

## Top 3 Opportunities
1. **Scale {top_cat} content** — highest engagement category. Increase publishing frequency by 20% and assign senior authors.
2. **Data-driven headlines** — articles with specific numbers consistently outperform. Enforce this in editorial guidelines.
3. **Author leverage** — top author significantly outperforms platform average. Pair them with underperforming categories.

## Underperforming Areas
**{bottom_cat}** has a {gap:.0f} point engagement gap vs the top category.
Action: Audit the last 30 {bottom_cat} articles. Compare headline structure, length, and publishing time vs top performers.

## Content Strategy for Next Week
Produce 3 long-form {top_cat} pieces targeting high search-volume topics.
Each piece should have a data-driven headline with specific numbers and at least one power word.

## Predicted Growth Lever
Improving {bottom_cat} engagement by 15% through headline optimization could add
approximately {int(bot_eng * 0.15 * 1250):,} engagement points to weekly platform totals.
This is the highest-leverage action available given the current content distribution.
"""
    return {"source": "rule_based", "content": content}


def _rule_based_headlines(category, topic):
    templates = [
        f"Breaking: {topic} Reshapes {category} Landscape",
        f"5 Reasons {topic} Is Changing {category} Forever",
        f"Exclusive: What {topic} Means for {category} in 2025",
        f"Is {topic} the Biggest {category} Story of the Year?",
        f"Record Numbers: How {topic} Is Driving {category} Growth",
    ]
    return {"source": "rule_based", "headlines": templates}
