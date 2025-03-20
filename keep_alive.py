from flask import Flask, request
from threading import Thread

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Bot is running."

def run():
    app.run(host='0.0.0.0', port=8888)

def keep_alive():
    t = Thread(target=run)
    t.start()
