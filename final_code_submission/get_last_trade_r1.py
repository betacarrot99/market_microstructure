import requests
import time
import logging
from datetime import datetime
import os
import csv

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
URL = 'https://fapi.binance.com'
METHOD = '/fapi/v1/trades'
SYMBOL = 'BTCUSDT'
CSV_FILE = 'result/live_data_agg.csv'
SAMPLE_INTERVAL = 0.2  # seconds (200ms)
AGGREGATION_PERIOD = 1  # seconds

# ─── SETUP ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s [%(levelname)-5s]  %(message)s',
    level=logging.INFO
)
# Ensure output directory exists
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

# ─── DATA FUNCTIONS ────────────────────────────────────────────────────────────
def get_last_trade_price(symbol):
    """
    Fetch the latest trade for the given symbol and return its price as float.
    """
    response = requests.get(URL + METHOD, params={'symbol': symbol})
    response.raise_for_status()
    trade = response.json()[0]
    return float(trade['price'])


def append_to_csv(timestamp, high, low, last, mean):
    """
    Append aggregated data to a CSV file, creating a header if necessary.
    """
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'high', 'low', 'price', 'mean'])
        writer.writerow([timestamp, high, low, last, mean])

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    while True:
        start_time = time.time()
        prices = []

        # Collect 200ms samples for one second
        while time.time() - start_time < AGGREGATION_PERIOD:
            try:
                price = get_last_trade_price(SYMBOL)
                prices.append(price)
            except Exception as e:
                logging.warning(f"Error fetching trade: {e}")
            time.sleep(SAMPLE_INTERVAL)

        # Perform aggregation if we have any data
        if prices:
            high_price = max(prices)
            low_price = min(prices)
            last_price = prices[-1]
            mean_price = sum(prices) / len(prices)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Log aggregated metrics
            logging.info(
                f"Aggregated @ {timestamp} | "
                f"High = {high_price:.2f} | Low = {low_price:.2f} | "
                f"LAST = {last_price:.2f} "
            )

            # Save to CSV
            append_to_csv(timestamp, high_price, low_price, last_price, mean_price)

        # Wait for the next aggregation period to start
        elapsed = time.time() - start_time
        if elapsed < AGGREGATION_PERIOD:
            time.sleep(AGGREGATION_PERIOD - elapsed)
