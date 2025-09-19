# AI News Sentiment Forecasting with Anomaly & Surge Detection  

This project analyzes **AI-related news articles** using NewsAPI, that performs **sentiment analysis**, and forecasts future sentiment trends with **Amazon Chronos (ChronosPipeline)** using **amazon/chronos-t5-tiny** model available via Hugging Face. 

It also detects **anomalies in sentiment trends**, **keyword surges**, and sends **Slack alerts** for important events.  

---

## Features  

-  Fetches latest AI-related news articles via **NewsAPI**  
-  Cleans and normalizes text with **ftfy + regex + unicodedata**  
-  Performs **sentiment analysis** using **TextBlob**  
-  Creates a structured dataset (`ai_news_sentiment_forecast.csv`)  
-  Forecasts **future sentiment trends** using **Amazon Chronos**  
-  Detects **anomalies** in sentiment (using Z-score method)  
-  Detects **keyword surges** in trending topics  
-  Sends **real-time Slack alerts** for anomalies, surges, or negative sentiment  
-  Generates **visual plots** of sentiment trends and forecasts  

---
