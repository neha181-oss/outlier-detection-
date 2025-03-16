import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import webbrowser
import ccxt

nltk.download('vader_lexicon')

def fetch_stock_data(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df["Returns"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

def detect_anomalies(df):
    df["Z-Score"] = zscore(df["Returns"])
    df["Z-Anomaly"] = df["Z-Score"].apply(lambda x: 1 if abs(x) > 2.5 else 0)
    
    iso_forest = IsolationForest(contamination=0.05)
    df["ISO-Anomaly"] = iso_forest.fit_predict(df[["Returns"]])
    df["ISO-Anomaly"] = df["ISO-Anomaly"].apply(lambda x: 1 if x == -1 else 0)
    
    return df

def fetch_news(ticker):
    API_KEY = "012df2997adc40a280739307043cc16c"
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}"
    response = requests.get(url).json()
    articles = [article["title"] for article in response.get("articles", [])[:5]]
    
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(title)["compound"] for title in articles]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    
    return avg_sentiment, articles

def send_outlook_email(subject, body, to_email):
    mailto_link = f"mailto:{to_email}?subject={subject}&body={body}"
    webbrowser.open(mailto_link)

def fetch_crypto_data(symbol):
    binance = ccxt.binance()
    ohlcv = binance.fetch_ohlcv(symbol, timeframe="1d", limit=180)
    df = pd.DataFrame(ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Returns"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

st.title("📈 Stock Anomaly Detector")
st.sidebar.header("Select Stock")

# User selects stock symbol
stock_symbol = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, TSLA)", "AAPL")

if st.sidebar.button("Analyze"):
    df = fetch_stock_data(stock_symbol)
    df = detect_anomalies(df)
    sentiment, news = fetch_news(stock_symbol)
    
    st.subheader("Stock Data & Anomalies")
    st.dataframe(df.tail(10))
    
    st.subheader("Anomaly Detection")
    st.write(df[df["Z-Anomaly"] == 1])
    
    st.subheader("News Sentiment Analysis")
    st.write(f"Average Sentiment Score: {sentiment}")
    st.write(news)
    
    if st.button("Send Email Alert"):
        send_outlook_email("Stock Anomaly Detected", f"{stock_symbol} has an unusual price movement!", "user@example.com")
        st.success("Email Alert Sent!")
