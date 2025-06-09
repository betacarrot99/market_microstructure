# live_trading/live_signals_individual.py
import os
import time
import logging
import pandas as pd
from datetime import datetime
from collections import defaultdict
from predictor.signal_manager import SignalManager
from signal_logic import evaluate_individual_predictions
from trade_fetcher import LiveTrade

# Config
FETCH_INTERVAL = 0.4  # seconds
PREDICT_INTERVAL = 2  # seconds
HORIZON = 2           # seconds for future evaluation
THRESHOLD = 0.000001
CSV_FILE = "live_signals_individual.csv"
PARAMS_FILE = "../data/best_signal_params.csv"

# Logging setup
logging.basicConfig(format='%(asctime)s [%(levelname)-5s] %(message)s', level=logging.INFO)

# Load tuned parameters
param_df = pd.read_csv(PARAMS_FILE)
param_dict = {
    row["Indicator"]: {
        k: int(v) if float(v).is_integer() else float(v)
        for k, v in row.drop(["Indicator"]).dropna().items()
    }
    for _, row in param_df.iterrows()
}

# Track signal accuracy
signal_accuracy = defaultdict(list)
pending_predictions = []

# Initialize signal manager and live feed
manager = SignalManager()
live = LiveTrade()

# Init CSV
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w") as f:
        f.write("timestamp,price," + ",".join(param_dict.keys()) + "\n")

# Loop
last_predict = time.monotonic()
while True:
    live.get_last_trade()
    df = live.get_dataframe()
    now = time.monotonic()

    # Evaluate pending
    evaluate_individual_predictions(df, now, pending_predictions, signal_accuracy, THRESHOLD)

    # Predict every interval
    if now - last_predict >= PREDICT_INTERVAL:
        signal_df = manager.generate_all_signals(df, param_dict)
        signals = signal_df.iloc[-1].to_dict()

        price_now = df["price"].iloc[-1]
        row = f"{datetime.utcnow().isoformat()},{price_now}," + ",".join(str(signals.get(k, "")) for k in param_dict.keys())
        with open(CSV_FILE, "a") as f:
            f.write(row + "\n")

        # Print with accuracy
        acc_strs = []
        for name in param_dict.keys():
            vals = signal_accuracy[name]  # now a list of True/False
            acc = f"{100 * sum(vals) / len(vals):.1f}%" if vals else "N/A"
            sig = signals.get(name, 0)
            acc_strs.append(f"{name}: {sig:+} ({acc})")

        logging.info(f"Price: {price_now:.2f} | " + " | ".join(acc_strs))

        # Track prediction
        pending_predictions.append({
            "timestamp": now,
            "eval_time": now + HORIZON,
            "price": price_now,
            "signals": signals
        })

        last_predict = now

    time.sleep(FETCH_INTERVAL)
