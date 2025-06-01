import pandas as pd
import time
import logging
from datetime import datetime

logging.basicConfig(format='%(asctime)s [%(levelname)-5s]  %(message)s', level=logging.INFO)

span_df = pd.read_csv("result/sma_param_log3.csv")
ewma_rows = span_df[span_df['method']=='EWMA']

CSV_FILE = 'result/live_data.csv'
SHORT_SPAN = int(ewma_rows['short_window'].values)
print(SHORT_SPAN)
LONG_SPAN = int(ewma_rows['long_window'].values)
print(LONG_SPAN)
ROLLING_WINDOW = max(SHORT_SPAN, LONG_SPAN)
CHECK_INTERVAL = 2  # seconds

def calculate_signal(df):
    df['short_ewma'] = df['price'].ewm(span=SHORT_SPAN, adjust=False).mean()
    df['long_ewma'] = df['price'].ewm(span=LONG_SPAN, adjust=False).mean()
    df['Signal'] = (df['short_ewma'] > df['long_ewma']).astype(int)
    return df

prev_signal = None

while True:
    try:
        df = pd.read_csv(CSV_FILE)
        if len(df) >= ROLLING_WINDOW:
            df = df.tail(ROLLING_WINDOW).reset_index(drop=True)
            df = calculate_signal(df)
            current_signal = df.iloc[-1]['Signal']
            # if current_signal != prev_signal:
            logging.info(f"New Signal: {'UP' if current_signal == 1 else 'DOWN'}")
            prev_signal = current_signal

    except Exception as e:
        logging.warning(f"Error reading or processing file: {e}")

    time.sleep(CHECK_INTERVAL)

