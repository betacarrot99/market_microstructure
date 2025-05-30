import pandas as pd
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(format='%(asctime)s [%(levelname)-5s]  %(message)s', level=logging.INFO)

# Load parameter CSV
span_df = pd.read_csv("result/sma_param_log3.csv")

# Filter for EWMA method (with robust string matching)
ewma_rows = span_df[span_df['method'].astype(str).str.strip().str.upper() == 'EWMA']

# Use the most recent EWMA row (last one)
latest_ewma_row = ewma_rows.iloc[-1]

# Extract span values
SHORT_SPAN = int(latest_ewma_row['short_window'])
LONG_SPAN = int(latest_ewma_row['long_window'])
print("SHORT_SPAN:", SHORT_SPAN)
print("LONG_SPAN:", LONG_SPAN)

# Constants
CSV_FILE = 'result/live_data.csv'
ROLLING_WINDOW = max(SHORT_SPAN, LONG_SPAN)
CHECK_INTERVAL = 2  # seconds

# EWMA signal calculation function
def calculate_signal(df):
    df['short_ewma'] = df['price'].ewm(span=SHORT_SPAN, adjust=False).mean()
    df['long_ewma'] = df['price'].ewm(span=LONG_SPAN, adjust=False).mean()
    df['Signal'] = (df['short_ewma'] > df['long_ewma']).astype(int)
    return df

# Main loop
prev_signal = None

while True:
    try:
        df = pd.read_csv(CSV_FILE)
        if len(df) >= ROLLING_WINDOW:
            df = df.tail(ROLLING_WINDOW).reset_index(drop=True)
            df = calculate_signal(df)
            current_signal = df.iloc[-1]['Signal']
            logging.info(f"New Signal: {'UP' if current_signal == 1 else 'DOWN'}")
            prev_signal = current_signal
    except Exception as e:
        logging.warning(f"Error reading or processing file: {e}")
    
    time.sleep(CHECK_INTERVAL)
