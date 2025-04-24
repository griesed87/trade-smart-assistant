# TradeSmart Assistant - With Sentiment Analysis and Trade Logging

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import plotly.graph_objects as go
from textblob import TextBlob
import requests

# ----- SETTINGS -----
st.set_page_config(page_title="TradeSmart Assistant", layout="wide")
st.title("📈 TradeSmart Assistant")

symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=0)
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=1)

# ----- GET DATA -----
data = yf.download(tickers=symbol, period=period, interval=interval)
data.dropna(inplace=True)

# ----- INDICATORS -----
def calculate_indicators(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

data = calculate_indicators(data)

# ----- NEWS SENTIMENT -----
def get_sentiment(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey=b09dc577aa20480bafaa61f6cfbeeb25"
    response = requests.get(url)
    if response.status_code != 200:
        return 0  # Neutral if error
    articles = response.json().get("articles", [])
    sentiments = [TextBlob(article['title']).sentiment.polarity for article in articles if article['title']]
    if sentiments:
        avg_sentiment = np.mean(sentiments)
        return avg_sentiment
    return 0

sentiment_score = get_sentiment(symbol)
st.sidebar.metric("Sentiment Score", f"{sentiment_score:.2f}")

# ----- SIGNALS -----
def generate_signals(df):
    df['Buy Signal'] = (df['MACD'] > df['Signal']) & (df['RSI'] < 70) & (sentiment_score > 0)
    df['Sell Signal'] = (df['MACD'] < df['Signal']) & (df['RSI'] > 30) & (sentiment_score < 0)
    return df

data = generate_signals(data)

# ----- PLOT CHART -----
def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA12'], mode='lines', name='EMA12'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA26'], mode='lines', name='EMA26'))
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', yaxis="y2"))

    buy_signals = df[df['Buy Signal']]
    sell_signals = df[df['Sell Signal']]

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                             mode='markers', marker=dict(symbol='triangle-up', size=10),
                             name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                             mode='markers', marker=dict(symbol='triangle-down', size=10),
                             name='Sell Signal'))

    fig.update_layout(title=f"{symbol} Price Chart with Indicators",
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

plot_chart(data)

# ----- SUMMARY -----
st.subheader("Latest Signal")
latest_signal = "Hold"
if data['Buy Signal'].iloc[-1]:
    latest_signal = "Buy"
elif data['Sell Signal'].iloc[-1]:
    latest_signal = "Sell"
st.metric(label="Signal", value=latest_signal)

# ----- TRADE LOGGING -----
log_entry = {
    'Time': dt.datetime.now(),
    'Symbol': symbol,
    'Signal': latest_signal,
    'Price': data['Close'].iloc[-1],
    'Sentiment': sentiment_score
}

if 'trade_log' not in st.session_state:
    st.session_state['trade_log'] = []

if st.button("Log This Trade"):
    st.session_state['trade_log'].append(log_entry)
    st.success("Trade logged!")

st.subheader("Trade Log")
st.dataframe(pd.DataFrame(st.session_state['trade_log']))
