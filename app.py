import subprocess
subprocess.run(["pip", "install", "yfinance", "pandas", "numpy", "scikit-learn", "scipy", "nltk", "requests", "ccxt", "torch", "transformers", "tensorflow"])

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import ccxt
import nltk
import torch
from transformers import pipeline
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense
from tf_keras.preprocessing.sequence import TimeseriesGenerator
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('punkt')

# Function to fetch stock data
def fetch_stock_data(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df["Returns"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

# Function to detect anomalies using multiple methods
def detect_anomalies(df):
    df["Z-Score"] = zscore(df["Returns"])
    df["Z-Anomaly"] = df["Z-Score"].apply(lambda x: 1 if abs(x) > 2.5 else 0)
    
    iso_forest = IsolationForest(contamination=0.05)
    df["ISO-Anomaly"] = iso_forest.fit_predict(df[["Returns"]])
    df["ISO-Anomaly"] = df["ISO-Anomaly"].apply(lambda x: 1 if x == -1 else 0)
    
    lof = LocalOutlierFactor(n_neighbors=20)
    df["LOF-Anomaly"] = lof.fit_predict(df[["Returns"]])
    df["LOF-Anomaly"] = df["LOF-Anomaly"].apply(lambda x: 1 if x == -1 else 0)
    
    oc_svm = OneClassSVM(nu=0.05)
    df["SVM-Anomaly"] = oc_svm.fit_predict(df[["Returns"]])
    df["SVM-Anomaly"] = df["SVM-Anomaly"].apply(lambda x: 1 if x == -1 else 0)
    
    return df

# Function to train LSTM model
def train_lstm(df):
    data = df["Returns"].values.reshape(-1, 1)
    generator = TimeseriesGenerator(data, data, length=5, batch_size=1)
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(5, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=5, verbose=0)
    
    predictions = model.predict(generator).flatten()
    df = df.iloc[5:].copy()
    df["LSTM-Anomaly"] = (abs(df["Returns"] - predictions) > 2 * np.std(predictions)).astype(int)
    return df

# Function to fetch news and perform sentiment analysis
def fetch_news(ticker):
    API_KEY = "012df2997adc40a280739307043cc16c"
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}"
    response = requests.get(url).json()
    articles = [article["title"] for article in response.get("articles", [])[:5]]
    
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(title)["compound"] for title in articles]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    
    return avg_sentiment, articles

# BERT-based sentiment analysis
def bert_sentiment_analysis(articles):
    sentiment_pipeline = pipeline("sentiment-analysis")
    scores = [sentiment_pipeline(article)[0]['score'] * (1 if sentiment_pipeline(article)[0]['label'] == 'POSITIVE' else -1) for article in articles]
    return np.mean(scores) if scores else 0

# Fetch crypto data
def fetch_crypto_data(symbol):
    binance = ccxt.binance()
    ohlcv = binance.fetch_ohlcv(symbol, timeframe="1d", limit=180)
    df = pd.DataFrame(ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Returns"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

# Main script execution (No Streamlit)
if __name__ == "__main__":
    stock_symbol = input("Enter stock ticker (e.g., AAPL, TSLA): ")
    
    print("\nFetching Stock Data...")
    df = fetch_stock_data(stock_symbol)
    print("\nDetecting Anomalies...")
    df = detect_anomalies(df)
    
    print("\nTraining LSTM Model...")
    df = train_lstm(df)
    
    print("\nFetching News Sentiment...")
    sentiment, news = fetch_news(stock_symbol)
    bert_sentiment = bert_sentiment_analysis(news)
    
    print("\n📊 **Stock Data & Anomalies (Last 10 Days)**")
    print(df.tail(10))
    
    print("\n🚨 **Anomaly Detection Results**")
    anomalies = df[df.iloc[:, -4:].sum(axis=1) >= 2]
    print(anomalies if not anomalies.empty else "No significant anomalies detected.")
    
    print("\n📰 **News Sentiment Analysis**")
    print(f"VADER Sentiment Score: {sentiment}")
    print(f"BERT Sentiment Score: {bert_sentiment}")
    print("Top News Headlines:")
    for article in news:
        print(f"- {article}")
