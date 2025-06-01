import pandas as pd
import time
import logging
from datetime import datetime
import joblib

logging.basicConfig(format='%(asctime)s [%(levelname)-5s]  %(message)s', level=logging.INFO)

# Load the latest logistic regression configuration
config_df = pd.read_csv("result/sma_param_log3.csv")
logreg_config = config_df[config_df['method'] == 'LogReg'].iloc[-1]
LAG = int(logreg_config['short_window'])

# Load trained logistic regression model
model = joblib.load("result/logreg_model.pkl")

CSV_FILE = 'result/live_data.csv'
ROLLING_WINDOW = LAG + 2  # enough for lag features and one-step-ahead accuracy
CHECK_INTERVAL = 2  # seconds

correct_predictions = 0
total_predictions = 0

def create_lag_features(df, lag):
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['price'].shift(i)
    return df

while True:
    try:
        df = pd.read_csv(CSV_FILE)
        if len(df) >= ROLLING_WINDOW:
            df = df.tail(ROLLING_WINDOW).reset_index(drop=True)
            df = create_lag_features(df, LAG).dropna()

            X_live = df[[f'lag_{i}' for i in range(1, LAG + 1)]]
            current_signal = model.predict(X_live.iloc[[-2]])[0]

            price_now = df.iloc[-2]['price']
            price_next = df.iloc[-1]['price']

            # Evaluate prediction
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
                f"Predicted: {'UP' if current_signal == 1 else 'DOWN'} | "
                f"Price Now: {price_now:.2f} → Next: {price_next:.2f} | "
                f"Result: {result} | Accuracy: {accuracy:.2f}%"
            )

    except Exception as e:
        logging.warning(f"Error reading or processing file: {e}")

    time.sleep(CHECK_INTERVAL)
