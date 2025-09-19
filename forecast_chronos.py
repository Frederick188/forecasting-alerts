import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from dotenv import load_dotenv
import requests


DATASET = "D:\\projects\\forcasting_alerts\\ai_news_sentiment_forecast.csv"
FORECAST_DAYS = 5
Z_THRESHOLD = 2        # For anomaly detection
SURGE_THRESHOLD = 3    # Minimum mentions for a keyword surge

# LOAD ENV
load_dotenv()
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")

#SLACK ALERT FUNCTION 
def send_slack_alert(message):
    if SLACK_WEBHOOK:
        try:
            requests.post(SLACK_WEBHOOK, json={"text": message})
        except Exception as e:
            print("Slack error:", e)

# ANOMALY DETECTION 
def detect_anomalies(daily_score, z_threshold=Z_THRESHOLD):
    anomalies = []
    y = daily_score["y"]
    mean, std = np.mean(y), np.std(y)
    for i, val in enumerate(y):
        z_score = (val - mean) / std if std > 0 else 0
        if abs(z_score) > z_threshold:
            anomalies.append((daily_score["ds"].iloc[i], val))
            send_slack_alert(f"[ANOMALY] 🚨 Unusual sentiment on {daily_score['ds'].iloc[i]}: {val:.2f}")
    
    if not anomalies:
        msg = "[INFO] ✅ No anomalies detected"
        print(msg)
        send_slack_alert(msg) 
    
    return anomalies

# FORECAST FUNCTION 
chronos = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny")

def forecast_scores(daily, days=FORECAST_DAYS):
    if len(daily) <= 2:
        return []

    context = torch.tensor(daily["y"].values, dtype=torch.float32)
    preds = chronos.predict(context=context, prediction_length=days)
    
    preds_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.array(preds)
    preds_np = preds_np[:, :days]  # Use only required days
    forecast_mean = [float(preds_np[:, i].mean()) for i in range(days)]
    return forecast_mean

# PLOT FORECAST 
def plot_forecast_separately(daily, forecast_vals, anomalies=[]):
    future_dates = pd.date_range(
        start=daily["ds"].iloc[-1] + pd.Timedelta(days=1),
        periods=len(forecast_vals),
        freq="D"
    )
    
    plt.figure(figsize=(10,5))
    plt.plot(daily["ds"], daily["y"], color="blue", marker="o", label="Actual")
    plt.plot(future_dates, forecast_vals, color="orange", marker="o", label=f"{FORECAST_DAYS}-Day Forecast")
    
    if anomalies:
        anomaly_dates, anomaly_vals = zip(*anomalies)
        plt.scatter(anomaly_dates, anomaly_vals, color="red", s=90, label="Anomalies")
    
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.title("AI News Sentiment Forecast with Chronos")
    plt.legend()
    plt.tight_layout()
    plt.show()

# KEYWORD SURGE DETECTION 
def detect_keyword_surge(df, surge_threshold=SURGE_THRESHOLD):
    keyword_counts = df.groupby(["date", "keyword"]).size().reset_index(name="count")
    if keyword_counts.empty:
        return
    latest_day = keyword_counts["date"].max()
    recent_day = keyword_counts[keyword_counts["date"] == latest_day]
    for _, row in recent_day.iterrows():
        if row["count"] >= surge_threshold:
            msg = f"[ALERT] 🔎 Keyword surge on {row['date']}: {row['count']} mentions of '{row['keyword']}'"
            print(msg)
            send_slack_alert(msg)

#  PIPELINE FUNCTION
def run_pipeline(data_set=DATASET, forecast_days=FORECAST_DAYS):
    
    df = pd.read_csv(data_set)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    for idx, row in df.iterrows():
        if row["score"] < 0:
            send_slack_alert(f"⚠️ Negative sentiment detected (score={row['score']:.2f}): {row['title'][:200]}")

    # Aggregate daily sentiment
    daily_score = df.groupby("date")["score"].mean().reset_index()
    daily_score.columns = ["ds", "y"]


    anomalies = detect_anomalies(daily_score)
    forecast_vals = forecast_scores(daily_score, days=forecast_days)
    plot_forecast_separately(daily_score, forecast_vals, anomalies)
    detect_keyword_surge(df)

    print(f"\nForecasted values for next {forecast_days} days:")
    for i, val in enumerate(forecast_vals, 1):
        print(f"Day {i}: {val:.4f}")

    return {
        "forecast": forecast_vals,
        "anomalies": anomalies
    }

# Run 
if __name__ == "__main__":
    run_pipeline()
