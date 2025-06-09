# live_trading/live_signals_aggregated.py
import os
import time
import logging
import pandas as pd
from datetime import datetime
from predictor.signal_manager import SignalManager
from trade_fetcher import LiveTrade
from predictor.aggregator.aggregator_manager import AggregatorManager
from collections import defaultdict

# Config
FETCH_INTERVAL = 0.4
PREDICT_INTERVAL = 2
CSV_FILE = "live_signals_aggregated.csv"
PARAMS_FILE = "../data/best_signal_params.csv"
AGGREGATORS = ["equal_weight", "accuracy_weight", "majority_vote"]
THRESHOLD = 0.000001
HORIZON = 2

# Logging
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

# Initialize
manager = SignalManager()
live = LiveTrade()
agg_manager = AggregatorManager(AGGREGATORS)
signal_accuracy = defaultdict(list)
agg_accuracy = defaultdict(list)
pending_predictions = []

# Init CSV
header_written = os.path.exists(CSV_FILE)
with open(CSV_FILE, "a") as f:
    if not header_written:
        base_cols = ["timestamp", "price"] + list(param_dict.keys())
        agg_cols = AGGREGATORS
        f.write(",".join(base_cols + agg_cols) + "\n")

# Evaluate predictions

def evaluate_predictions(df, now, prediction_queue, signal_accuracy, agg_accuracy, threshold=0.000001):
    current_price = df["price"].iloc[-1]
    updated = []

    for pred in prediction_queue:
        if now >= pred["eval_time"]:
            threshold_val = pred["price"] * threshold
            price_diff = current_price - pred["price"]

            if price_diff > threshold_val:
                truth = 1
            elif price_diff < -threshold_val:
                truth = -1
            else:
                truth = 0

            for name, val in pred["signals"].items():
                signal_accuracy[name].append(val == truth)

            for name, val in pred["aggregated"].items():
                agg_accuracy[name].append(val == truth)
        else:
            updated.append(pred)

    prediction_queue.clear()
    prediction_queue.extend(updated)

# Loop
last_predict = time.monotonic()
while True:
    live.get_last_trade()
    df = live.get_dataframe()
    now = time.monotonic()

    evaluate_predictions(df, now, pending_predictions, signal_accuracy, agg_accuracy, THRESHOLD)

    if now - last_predict >= PREDICT_INTERVAL:
        signal_df = manager.generate_all_signals(df, param_dict)
        signals = signal_df.iloc[-1].to_dict()

        price_now = df["price"].iloc[-1]
        agg_results = agg_manager.aggregate(signals)

        # Round outputs to nearest discrete signal
        rounded_signals = {k: int(round(v)) if pd.notna(v) else v for k, v in signals.items()}
        rounded_results = {k: int(round(v)) if pd.notna(v) else v for k, v in agg_results.items()}

        row_dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "price": price_now,
            **rounded_signals,
            **rounded_results,
        }
        df_out = pd.DataFrame([row_dict])
        df_out.to_csv(CSV_FILE, mode="a", index=False, header=False)

        # Accuracy logging for individuals
        individual_msg = []
        for k, v in rounded_signals.items():
            acc_list = signal_accuracy[k]
            acc = (100 * sum(acc_list) / len(acc_list)) if acc_list else None
            acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
            individual_msg.append(f"{k}: {v:+d} ({acc_str})") if pd.notna(v) else None

        # Accuracy logging for aggregators
        agg_msg = []
        for k, v in rounded_results.items():
            acc_list = agg_accuracy[k]
            acc = (100 * sum(acc_list) / len(acc_list)) if acc_list else None
            acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
            agg_msg.append(f"{k}: {v:+d} ({acc_str})") if pd.notna(v) else None

        logging.info(f"Price: {price_now:.2f} | INDIVIDUAL -> {' | '.join(individual_msg)} | AGGREGATED -> {' | '.join(agg_msg)}")

        # Save for future accuracy eval
        pending_predictions.append({
            "timestamp": now,
            "eval_time": now + HORIZON,
            "price": price_now,
            "signals": rounded_signals,
            "aggregated": rounded_results
        })

        last_predict = now

    time.sleep(FETCH_INTERVAL)
