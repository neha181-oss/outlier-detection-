import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import webbrowser
def fetch_stock_data(ticker,period="6mo"):
    stock=yf.Ticker(ticker)
    df=stock.history(period=period)
    df["Returns"]=df["Close"].pct_change()
    df.dropna(inplace=True)
    return df
df=fetch_stock_data("AAPL")
print(df.head())

def detect_anamolies(df):
    df["Z-Score"]= zscore(df["Returns"])
    df["Z-Anamoly"]=df["Z-Score"].apply(lambda x:1 if abs(x)>2.5 else 0) #flagging anamolies

    iso_forest= IsolationForest(contamination=0.05)#Train isolation model
    df["ISO-anamoly"]= iso_forest.fit_predict(df[["Returns"]]) #predict anamolies
    df["ISO-anamoly"]= df["ISO-anamoly"].apply( lambda x:1 if x== -1 else 0) #convert to binary flag 
    return df
df=detect_anamolies(df)
print(df[df["Z-Anamoly"]==1]) #show anamolies detected by Z-score Method 

import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
def fetch_news(ticker):
    API_KEY = "012df2997adc40a280739307043cc16c"
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}"
    response = requests.get(url).json()
    articles = [article["title"] for article in response["articles"][:5]]  # Get top 5 news titles

    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(title)["compound"] for title in articles]  # Analyze sentiment
    avg_sentiment = np.mean(sentiments)  # Compute average sentiment score
    
    return avg_sentiment, articles

sentiment, news = fetch_news("AAPL")  # Fetch Apple news sentiment
print(f"News Sentiment: {sentiment}")
print(news)

# Outlook Email Alert (No Password Needed)
def send_outlook_email(subject, body, to_email):
    mailto_link = f"mailto:{to_email}?subject={subject}&body={body}"
    webbrowser.open(mailto_link)

send_outlook_email("Stock Anomaly Detected", "AAPL has an unusual price movement!", "user@example.com")
