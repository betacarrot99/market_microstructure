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

# Save path: ~/Documents/QF635/live_data.csv
CSV_FILE = os.path.expanduser("/market_microstructure/live_data.csv")

# Create the folder if it doesn't exist
output_dir = os.path.dirname(CSV_FILE)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Create the file with headers if not exist
if not os.path.isfile(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'price', 'qty'])

# Track last printed second to avoid duplicates
last_printed_second = None

def on_message(ws, message):
    global last_printed_second

    data = json.loads(message)
    price = float(data['p'])
    qty = float(data['q'])
    now = datetime.now()
    current_second = now.replace(microsecond=0)

    # Only log one trade per second
    if last_printed_second != current_second:
        last_printed_second = current_second
        timestamp_str = current_second.strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Trade: Price = {price}, Qty = {qty:.3f}")

        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp_str, price, qty])

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

# Start the WebSocket connection
ws_url = "wss://fstream.binance.com/ws"
websocket.enableTrace(False)
ws = websocket.WebSocketApp(ws_url,
                             on_message=on_message,
                             on_error=on_error,
                             on_close=on_close,
                             on_open=on_open)

ws.run_forever()