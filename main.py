import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

def fetch_stock_data(ticker,period="6mo"):
    stock=yf.Ticker(ticker)
    df=stock.history(period=period)
    df["Returns"]=df["Close"].pct_change()
    df.dropna(inplace=True)
    return df
df=fetch_stock_data("AAPL")
print(df.head())