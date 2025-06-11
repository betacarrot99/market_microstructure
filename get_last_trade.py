import websocket
import json
import logging
import csv
import os
from datetime import datetime
import time

# Setup logging
logging.basicConfig(
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

CSV_FILE = os.path.join(os.path.dirname(__file__), "live_data.csv")

output_dir = os.path.dirname(CSV_FILE)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Create the file with headers if not exist
if not os.path.isfile(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'low_price', 'low_qty', 'high_price', 'high_qty', 'close_price'])

# Buffer for trades in the current second
trade_buffer = []
current_second = None

def summarize_and_save(second, trades):
    if not trades:
        return

    # Extract price and quantity arrays
    prices = [t['price'] for t in trades]
    qtys = [t['qty'] for t in trades]

    # High/low price and total qty at those prices
    low_price = min(prices)
    high_price = max(prices)
    low_qty = sum(qty for p, qty in zip(prices, qtys) if p == low_price)
    high_qty = sum(qty for p, qty in zip(prices, qtys) if p == high_price)

    # Close = last trade of the second
    close_price = prices[-1]

    timestamp_str = second.strftime('%Y-%m-%d %H:%M:%S')
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp_str, low_price, low_qty, high_price, high_qty, close_price])

    logging.info(f"{timestamp_str} | Low = {low_price} ({low_qty}), High = {high_price} ({high_qty}), Close = {close_price}")

def on_message(ws, message):
    global trade_buffer, current_second

    data = json.loads(message)
    price = float(data['p'])
    qty = float(data['q'])
    now = datetime.now().replace(microsecond=0)

    if current_second is None:
        current_second = now

    if now != current_second:
        summarize_and_save(current_second, trade_buffer)
        trade_buffer = []
        current_second = now

    trade_buffer.append({'price': price, 'qty': qty})

def on_error(ws, error):
    logging.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    logging.info("WebSocket closed")

def on_open(ws):
    logging.info("WebSocket opened")
    payload = {
        "method": "SUBSCRIBE",
        "params": ["btcusdt@trade"],
        "id": 1
    }
    ws.send(json.dumps(payload))

# Start WebSocket
ws_url = "wss://fstream.binance.com/ws"
websocket.enableTrace(False)
ws = websocket.WebSocketApp(ws_url,
                             on_message=on_message,
                             on_error=on_error,
                             on_close=on_close,
                             on_open=on_open)

ws.run_forever()
