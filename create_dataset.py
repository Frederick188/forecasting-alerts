import os
import re
import pandas as pd
import unicodedata
import ftfy
from dotenv import load_dotenv
from newsapi import NewsApiClient
from textblob import TextBlob

# Load API Keys
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Initialize News API Client
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# Text Cleaning 
def clean_text(text):
    if not text:
        return ""
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

#  Sentiment Analysis 
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 to +1
    if polarity > 0.05:
        sentiment = "Positive"
    elif polarity < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, polarity

#  Fetch News 
def fetch_news(keywords, limit=20, pages=3):
    all_articles = []
    for kw in keywords:
        for page in range(1, pages + 1):
            articles = newsapi.get_everything(
                q=kw,
                language="en",
                sort_by="publishedAt",
                page_size=limit,
                page=page
            ).get("articles", [])

            for a in articles:
                text = clean_text((a.get("title") or "") + " " + (a.get("description") or ""))
                sentiment, score = analyze_sentiment(text)
                
                all_articles.append({
                    "keyword": kw,
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", "unknown"),
                    "text": text,
                    "date": a.get("publishedAt", ""),
                    "sentiment": sentiment,
                    "score": score
                })
    return all_articles

# Create Dataset 
def create_dataset(filename="ai_news_sentiment_forecast.csv", limit=20, pages=3):
    keywords = [
        "Artificial Intelligence", "Machine Learning", "Deep Learning",
        "Neural Networks", "NLP", "Generative AI", "Computer Vision",
        "AI Ethics", "Chatbots", "Robotics"
    ]

    data = fetch_news(keywords, limit=limit, pages=pages)
    df = pd.DataFrame(data)
    
    # Keep only required columns
    df = df[["date", "keyword", "source", "title", "text", "sentiment", "score"]]
    
    # Save CSV
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Dataset saved as {filename}")
    return df

# Pipeline Function 
def pipeline(filename="ai_news_sentiment_forecast.csv", limit=20, pages=3):
    print("Starting pipeline...")
    df = create_dataset(filename=filename, limit=limit, pages=pages)
    print(f"Pipeline complete. {len(df)} rows saved.")
    return df

# Run 
if __name__ == "__main__":
    pipeline(limit=20, pages=3)
