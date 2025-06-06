import pandas as pd
import time
import logging
from datetime import datetime

logging.basicConfig(format='%(asctime)s [%(levelname)-5s]  %(message)s', level=logging.INFO)

# Load and filter for latest TSMOM config
span_df = pd.read_csv("result/sma_param_log3.csv")
tsmom_rows = span_df[span_df['method'] == 'TSMOM']
latest_tsmom = tsmom_rows.iloc[-1]

CSV_FILE = 'result/live_data.csv'
LOOKBACK = int(latest_tsmom['short_window'])  # Using short_window as the lookback period
ROLLING_WINDOW = LOOKBACK + 1
CHECK_INTERVAL = 2  # seconds

correct_predictions = 0
total_predictions = 0
prev_signal = None

def calculate_signal(df):
    df['past_price'] = df['price'].shift(LOOKBACK)
    df['Signal'] = (df['price'] > df['past_price']).astype(int)
    return df

while True:
    try:
        df = pd.read_csv(CSV_FILE)
        if len(df) >= ROLLING_WINDOW + 1:
            df = df.tail(ROLLING_WINDOW + 1).reset_index(drop=True)
            df = calculate_signal(df)

            current_signal = df.iloc[-2]['Signal']
            price_now = df.iloc[-2]['price']
            price_next = df.iloc[-1]['price']

            # Evaluate correctness of the signal
            is_correct = (
                (current_signal == 1 and price_next > price_now) or
                (current_signal == 0 and price_next < price_now)
            )

            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            accuracy = 100 * correct_predictions / total_predictions

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
