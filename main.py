import os
import time
import requests
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==== CONFIG ====
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

features = ['RSI', 'EMA9', 'EMA21', 'MACD', 'MACD_Signal']
assets = [
    'BTC-USD', 'ETH-USD', 'SOL-USD',
    'TSLA', 'NVDA', 'META', 'AAPL',
    'GC=F', 'CL=F',  # Gold, Oil
    'GBPJPY=X', 'EURUSD=X', 'USDJPY=X'  # Forex
]

interval_minutes = 30  # check every 30 mins

# ==== FUNCTIONS ====

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    requests.post(url, data=data)

def fetch_data(ticker):
    long_assets = ['GC=F', 'CL=F', 'USOIL']
    pe

