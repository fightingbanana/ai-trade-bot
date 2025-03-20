import os
import time
import json
import requests
import yfinance as yf
import pandas as pd
from flask import Flask, request
from threading import Thread
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keep_alive import keep_alive

# ==== CONFIG ====
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
features = ['RSI', 'EMA9', 'EMA21', 'MACD', 'MACD_Signal']
assets = {
    'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD'],
    'stocks': ['TSLA', 'NVDA', 'META', 'AAPL'],
    'commodities': ['GC=F', 'CL=F'],
    'forex': ['GBPJPY=X', 'EURUSD=X', 'USDJPY=X']
}
interval_minutes = 30
models = {}

# ==== FLASK ====
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Bot is running."

@app.route(f"/{TELEGRAM_BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    data = request.get_json()
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")
        if text:
            handle_command(text, chat_id)
    elif "callback_query" in data:
        callback = data["callback_query"]
        chat_id = callback["message"]["chat"]["id"]
        data_clicked = callback["data"]
        handle_command(data_clicked, chat_id)
    return "ok", 200

def send_telegram_message(message, chat_id=TELEGRAM_CHAT_ID, buttons=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message}
    if buttons:
        payload['reply_markup'] = json.dumps({"inline_keyboard": buttons})
    requests.post(url, data=payload)

# ==== TRADING LOGIC ====
def fetch_data(ticker):
    long_assets = ['GC=F', 'CL=F']
    period = '30d' if ticker in long_assets else '7d'
    df = yf.download(ticker, interval='15m', period=period, auto_adjust=True)
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df.dropna()

def compute_indicators(df):
    df['EMA9'] = df['close'].ewm(span=9).mean()
    df['EMA21'] = df['close'].ewm(span=21).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['close'].ewm(12).mean() - df['close'].ewm(26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(9).mean()
    return df

def label_data(df):
    df['Future_Close'] = df['close'].shift(-3)
    df.dropna(inplace=True)
    df['Target'] = (df['Future_Close'] > df['close']).astype(int)
    return df

def train_model_for(ticker):
    df = fetch_data(ticker)
    if df.empty:
        return None, None
    df = compute_indicators(df)
    df = label_data(df)
    X = df[features]
    y = df['Target']
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model, df

def get_prediction(model, df):
    latest = df.iloc[-1:]
    confidence_raw = model.predict_proba(latest[features])[0][1]
    signal = "✅ BUY" if confidence_raw >= 0.5 else "❌ SELL"
    direction = "📈 UP" if confidence_raw >= 0.5 else "📉 DOWN"
    confidence_final = confidence_raw if confidence_raw >= 0.5 else 1 - confidence_raw
    rsi = round(latest['RSI'].values[0], 2)
    macd = round(latest['MACD'].values[0], 4)
    macd_signal = round(latest['MACD_Signal'].values[0], 4)
    price = round(latest['close'].values[0], 2)
    trend = "📈 Bullish" if macd > macd_signal else "📉 Bearish"
    return f"""
📊 Signal: {signal}
Direction: {direction}
Model Confidence: {confidence_final:.0%}
Price: ${price:,.2f}
Trend: {trend}
RSI: {rsi}
"""

# ==== COMMAND HANDLER ====
def handle_command(text, chat_id):
    text = text.lower().strip()

    if text in ["/start", "/help"]:
        send_telegram_message(
            "🤖 Choose an asset category:",
            chat_id,
            buttons=[
                [{"text": "💸 Crypto", "callback_data": "/crypto"}],
                [{"text": "📈 Stocks", "callback_data": "/stocks"}],
                [{"text": "🛢️ Commodities", "callback_data": "/commodities"}],
                [{"text": "ℹ️ Info (type /info <symbol>)", "callback_data": "/help"}],
            ],
        )
        return

    if text.startswith("/crypto") or text == "crypto":
        symbols = assets['crypto']
    elif text.startswith("/stocks") or text == "stocks":
        symbols = assets['stocks']
    elif text.startswith("/commodities") or text == "commodities":
        symbols = assets['commodities']
    elif text.startswith("/info"):
        parts = text.split()
        if len(parts) == 2:
            symbol = parts[1].upper()
            model, df = train_model_for(symbol)
            if model:
                msg = f"📊 {symbol}\n{get_prediction(model, df)}"
            else:
                msg = f"⚠️ No data for {symbol}"
            send_telegram_message(msg, chat_id)
        else:
            send_telegram_message("⚠️ Usage: /info <symbol>", chat_id)
        return
    else:
        send_telegram_message("⚠️ Unknown command. Type /start to see options.", chat_id)
        return

    for symbol in symbols:
        model, df = train_model_for(symbol)
        if model:
            msg = f"📊 {symbol}\n{get_prediction(model, df)}"
        else:
            msg = f"⚠️ No data for {symbol}"
        send_telegram_message(msg, chat_id)

# ==== BACKGROUND WORKER ====
def background_refresh():
    while True:
        print("🔄 Background refresh running...")
        for category in assets:
            for symbol in assets[category]:
                try:
                    model, df = train_model_for(symbol)
                    if model:
                        models[symbol] = (model, df)
                        print(f"✅ Updated model for {symbol}")
                except Exception as e:
                    print(f"⚠️ Error updating {symbol}: {e}")
        time.sleep(interval_minutes * 60)

# ==== RUN ====
keep_alive()
Thread(target=background_refresh).start()
