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
    'GC=F', 'CL=F',
    'GBPJPY=X', 'EURUSD=X', 'USDJPY=X'
]

interval_minutes = 30

print("✅ Bot has started successfully.")

# ==== FUNCTIONS ====

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"⚠️ Telegram Error: {response.text}")
    except Exception as e:
        print(f"❌ Telegram Failed: {e}")

def fetch_data(ticker):
    long_assets = ['GC=F', 'CL=F', 'USOIL']
    period = '30d' if ticker in long_assets else '7d'
    df = yf.download(ticker, interval='15m', period=period, auto_adjust=True)
    if df.empty or len(df) < 100:
        print(f"⚠️ No data for {ticker}")
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

def train_model(df):
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def get_signal(model, df):
    latest = df.iloc[-1:]
    return "✅ BUY" if model.predict(latest[features])[0] == 1 else "❌ SELL"

# ==== MAIN LOOP ====
while True:
    print(f"🔁 Checking signals at {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    for asset in assets:
        try:
            df = fetch_data(asset)
            if df.empty:
                continue
            df = compute_indicators(df)
            df = label_data(df)
            model = train_model(df)
            signal = get_signal(model, df)

            latest_price = df['close'].iloc[-1]
            rsi = round(df['RSI'].iloc[-1], 2)
            macd = round(df['MACD'].iloc[-1], 4)
            macd_signal = round(df['MACD_Signal'].iloc[-1], 4)

            confidence_raw = model.predict_proba(df.iloc[[-1]][features])[0][1]
            direction = "📈 UP" if confidence_raw >= 0.5 else "📉 DOWN"
            confidence_final = confidence_raw if confidence_raw >= 0.5 else 1 - confidence_raw
            trend = "📈 Bullish" if macd > macd_signal else "📉 Bearish"

            message = f"""
📊 Signal: {signal}
Asset: {asset}
Direction: {direction}
Model Confidence: {confidence_final:.0%}
Price: ${latest_price:,.2f}
Trend: {trend}
RSI: {rsi}
📅 Timestamp: {time.strftime('%Y-%m-%d %H:%M UTC')}
"""
            print(message.strip())
            send_telegram_message(message.strip())

        except Exception as e:
            error_msg = f"⚠️ Error with {asset}: {e}"
            print(error_msg)
            send_telegram_message(error_msg)

    print(f"🕒 Sleeping for {interval_minutes} minutes...\n")
    time.sleep(interval_minutes * 60)

