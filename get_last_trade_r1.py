import websocket
import json
import logging
import csv
import os
from datetime import datetime

# Setup logging
logging.basicConfig(format='%(asctime)s [%(levelname)-5.5s]  %(message)s', level=logging.INFO)

# Output file path
CSV_FILE = "result/live_data.csv"

# Create CSV file if it doesn't exist
if not os.path.isfile(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'price', 'qty'])


# Define WebSocket message handler
def on_message(ws, message):
    data = json.loads(message)
    price = data['p']
    qty = data['q']
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    logging.info('Trade: Price = {}, Qty = {}'.format(price, qty))

    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, price, qty])


# Define WebSocket error handler
def on_error(ws, error):
    logging.error("WebSocket Error: {}".format(error))


# Define WebSocket close handler
def on_close(ws, close_status_code, close_msg):
    logging.info("WebSocket closed")


# Define WebSocket open handler
def on_open(ws):
    logging.info("WebSocket opened")
    payload = {
        "method": "SUBSCRIBE",
        "params": [
            "btcusdt@trade"
        ],
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
