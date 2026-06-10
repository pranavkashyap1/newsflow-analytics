import os
from groq import Groq


def get_model():
    try:
        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            return None

        return Groq(api_key=api_key)

    except Exception:
        return None


def generate_editorial_recommendations(kpis, category_df, top_articles_df):
    model = get_model()

    if model is None:
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

Provide exactly:

1. EXECUTIVE SUMMARY
2. TOP 3 OPPORTUNITIES
3. UNDERPERFORMING AREAS
4. CONTENT STRATEGY RECOMMENDATION
5. PREDICTED GROWTH LEVER

Be concise, data-driven, and actionable.
"""

    try:
        response = model.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return {
            "source": "groq",
            "content": response.choices[0].message.content
        }

    except Exception as e:
        print(f"Groq Error: {e}")
        return _rule_based_recommendations(kpis, category_df)


def generate_headline_suggestions(category, topic, performance_data):
    model = get_model()

    if model is None:
        return _rule_based_headlines(category, topic)

    prompt = f"""
You are an expert digital news editor.

Generate 5 highly engaging news headlines.

Category: {category}
Topic: {topic}

Rules:
- Under 15 words
- News-style
- Mix curiosity, numbers, and authority
- Suitable for a major publication
- Return ONLY the headlines as a numbered list
"""

    try:
        response = model.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.9,
            max_tokens=300
        )

        text = response.choices[0].message.content

        lines = [
            l.strip()
            for l in text.strip().split("\n")
            if l.strip()
        ]

        headlines = []

        for line in lines:
            cleaned = line.lstrip("0123456789.-) ").strip()
            if cleaned:
                headlines.append(cleaned)

        return {
            "source": "groq",
            "headlines": headlines[:5]
        }

    except Exception as e:
        print(f"Groq Error: {e}")
        return _rule_based_headlines(category, topic)


def _rule_based_recommendations(kpis, category_df):
    top_cat = kpis["top_category"]
    total_views = kpis["total_views"]
    avg_eng = kpis["avg_engagement"]

    sorted_df = category_df.sort_values(
        "avg_engagement",
        ascending=False
    )

    bottom_cat = sorted_df.iloc[-1]["category"]
    top_eng = sorted_df.iloc[0]["avg_engagement"]
    bot_eng = sorted_df.iloc[-1]["avg_engagement"]

    gap = top_eng - bot_eng

    content = f"""
## Executive Summary

{top_cat} content is the strongest performer with the highest average engagement score.

Total platform views stand at {total_views:,} with an average engagement score of {avg_eng:,.0f}.

## Top 3 Opportunities

1. Scale {top_cat} content.
2. Improve headline optimization.
3. Increase exposure for high-performing authors.

## Underperforming Areas

{bottom_cat} trails the leading category by {gap:.0f} engagement points.

## Content Strategy Recommendation

Publish more high-performing content formats and strengthen distribution.

## Predicted Growth Lever

Improving {bottom_cat} engagement could significantly increase total platform performance.
"""

    return {
        "source": "rule_based",
        "content": content
    }


def _rule_based_headlines(category, topic):
    headlines = [
        f"Breaking: {topic} Reshapes {category} Landscape",
        f"5 Reasons {topic} Is Changing {category} Forever",
        f"Exclusive: What {topic} Means for {category}",
        f"Is {topic} the Biggest {category} Story This Year?",
        f"How {topic} Could Transform {category} Markets"
    ]

    return {
        "source": "rule_based",
        "headlines": headlines
    }