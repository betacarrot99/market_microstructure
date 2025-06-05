import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime

logging.basicConfig(format='%(asctime)s [%(levelname)-5s]  %(message)s', level=logging.INFO)

# ─── Load the best‐tuned mean‐reversion parameters from the CSV log ─────────────
span_df = pd.read_csv("result/sma_param_log3.csv")
mr_rows = span_df[span_df['method'] == 'MeanReversion']

MEAN_SPAN = int(mr_rows['long_window'].values[0])
ROLLING_WINDOW = MEAN_SPAN

CSV_FILE = 'result/live_data.csv'
CHECK_INTERVAL = 2  # seconds

correct_predictions = 0
total_predictions = 0
prev_signal = None

def calculate_signal(df):
    """
    Given a DataFrame with a 'price' column, compute:
      - rolling_mean over MEAN_SPAN
      - Signal = 1 if price < rolling_mean (mean reversion buy), else 0 (flat)
    """
    df['rolling_mean'] = df['price'].rolling(window=MEAN_SPAN).mean()
    df['Signal'] = np.where(df['price'] < df['rolling_mean'], 1, 0)
    return df

while True:
    try:
        df = pd.read_csv(CSV_FILE)
        if len(df) >= ROLLING_WINDOW + 1:
            df = df.tail(ROLLING_WINDOW + 1).reset_index(drop=True)
            df = calculate_signal(df)

            # Use the second‐to‐last row’s signal to predict the last row’s direction:
            current_signal = int(df.iloc[-2]['Signal'])
            price_now = df.iloc[-2]['price']
            price_next = df.iloc[-1]['price']

            # Check if the prediction was correct:
            is_correct = (
                (current_signal == 1 and price_next > price_now) or
                (current_signal == 0 and price_next < price_now)
            )

            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            accuracy = 100 * correct_predictions / total_predictions
            print(MEAN_SPAN)
            result = "correct" if is_correct else "wrong"
            logging.info(
                f"New Signal: {'UP' if current_signal == 1 else 'DOWN'} | "
                f"Price Now: {price_now:.2f} → Next: {price_next:.2f} | "
                f"Result: {result} | Accuracy: {accuracy:.2f}%"
            )

            prev_signal = current_signal

    except Exception as e:
        logging.warning(f"Error reading or processing file: {e}")

    time.sleep(CHECK_INTERVAL)
