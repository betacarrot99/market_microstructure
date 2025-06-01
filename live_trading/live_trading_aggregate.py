import os
import time
import logging
import pandas as pd
import numpy as np
import requests
from collections import defaultdict
from datetime import datetime

# Logging setup
logging.basicConfig(format='%(asctime)s [%(levelname)-5s] %(message)s', level=logging.INFO)

# Config
TRADE_URL = 'https://fapi.binance.com/fapi/v1/trades'
SYMBOL = "BTCUSDT"
INTERVAL = 0.1   # prediction interval in seconds
HORIZON = 0.1    # evaluation horizon in seconds
MAXLEN = 1000    # data buffer size
CSV_FILE = "live_predictions.csv"
THRESHOLD = 0.00001  # 0.001% threshold

# Load best parameters => from backtesting
param_df = pd.read_csv("best_signal_params.csv").set_index("Indicator")

def get_params(indicator):
    row = param_df.loc[indicator]
    return {k: int(v) if float(v).is_integer() else float(v) for k, v in row.drop(["Train Acc", "Test Acc"]).dropna().items()}

# Get live trade
class LiveTrade:
    def __init__(self, symbol=SYMBOL, maxlen=MAXLEN):
        self.symbol = symbol
        self.maxlen = maxlen
        self.prices, self.timestamps, self.volumes = [], [], []

    def get_last_trade(self):
        try:
            response = requests.get(TRADE_URL, params={'symbol': self.symbol}, timeout=5)
            trade = response.json()[-1]
            self.timestamps.append(trade['time'])
            self.prices.append(float(trade['price']))
            self.volumes.append(float(trade['qty']))
            self.timestamps = self.timestamps[-self.maxlen:]
            self.prices = self.prices[-self.maxlen:]
            self.volumes = self.volumes[-self.maxlen:]
        except Exception as e:
            logging.error(f"Trade fetch error: {e}")

    def get_dataframe(self):
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "price": self.prices,
            "volume": self.volumes
        })

# Signals
def compute_signals(df):
    signals = {}
    try:
        if "RSI" in param_df.index:
            p = get_params("RSI")
            if len(df) >= p["RSI_period"]:
                delta = df["price"].diff()
                gain = delta.clip(lower=0).rolling(int(p["RSI_period"])).mean()
                loss = -delta.clip(upper=0).rolling(int(p["RSI_period"])).mean()
                rs = gain / loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                last_rsi = rsi.iloc[-1]
                signals["RSI"] = 1 if last_rsi < 30 else -1 if last_rsi > 70 else 0

        if "MACD" in param_df.index:
            p = get_params("MACD")
            min_len = max(p["MACD_fast"], p["MACD_slow"], p["MACD_signal"])
            if len(df) >= min_len:
                macd = df["price"].ewm(span=p["MACD_fast"]).mean() - df["price"].ewm(span=p["MACD_slow"]).mean()
                macd_sig = macd.ewm(span=p["MACD_signal"]).mean()
                signal_val = macd.iloc[-1] - macd_sig.iloc[-1]
                signals["MACD"] = 1 if signal_val > 0 else -1 if signal_val < 0 else 0

        if "Stoch" in param_df.index:
            p = get_params("Stoch")
            if len(df) >= p["Stoch_window"] + p["Stoch_smooth_k"]:
                low = df["price"].rolling(int(p["Stoch_window"])).min()
                high = df["price"].rolling(int(p["Stoch_window"])).max()
                k = 100 * (df["price"] - low) / (high - low)
                d = k.rolling(int(p["Stoch_smooth_k"])).mean()
                signal_val = k.iloc[-1] - d.iloc[-1]
                signals["Stoch"] = 1 if signal_val > 0 else -1 if signal_val < 0 else 0

        if "BB" in param_df.index:
            p = get_params("BB")
            if len(df) >= p["BB_window"]:
                mid = df["price"].rolling(int(p["BB_window"])).mean()
                std = df["price"].rolling(int(p["BB_window"])).std()
                upper = mid + float(p["BB_std_mult"]) * std
                lower = mid - float(p["BB_std_mult"]) * std
                price = df["price"].iloc[-1]
                if price < lower.iloc[-1]:
                    signals["BB"] = 1
                elif price > upper.iloc[-1]:
                    signals["BB"] = -1
                else:
                    signals["BB"] = 0

        if "OBV" in param_df.index:
            p = get_params("OBV")
            if len(df) >= p["OBV_ma"]:
                obv = np.where(df["price"].diff() > 0, df["volume"],
                               np.where(df["price"].diff() < 0, -df["volume"], 0)).cumsum()
                obv_series = pd.Series(obv, index=df.index)
                obv_ma = obv_series.rolling(int(p["OBV_ma"])).mean()
                signal_val = obv_series.iloc[-1] - obv_ma.iloc[-1]
                signals["OBV"] = 1 if signal_val > 0 else -1 if signal_val < 0 else 0

        if "VWAP" in param_df.index:
            p = get_params("VWAP")
            if len(df) >= p["VWAP_window"]:
                vwap = (df["price"] * df["volume"]).rolling(int(p["VWAP_window"])).sum() / df["volume"].rolling(int(p["VWAP_window"])).sum()
                signal_val = df["price"].iloc[-1] - vwap.iloc[-1]
                signals["VWAP"] = 1 if signal_val > 0 else -1 if signal_val < 0 else 0

        if "TWAP" in param_df.index:
            p = get_params("TWAP")
            if len(df) >= p["TWAP_window"]:
                twap = df["price"].rolling(int(p["TWAP_window"])).mean()
                signal_val = df["price"].iloc[-1] - twap.iloc[-1]
                signals["TWAP"] = 1 if signal_val > 0 else -1 if signal_val < 0 else 0

    except Exception as e:
        logging.warning(f"Signal computation error: {e}")
        signals = {k: 0 for k in param_df.index}

    return signals

def equal_weighted(signals: dict):
    return int(np.sign(np.mean([np.sign(v) for v in signals.values()]))) if signals else 0

def accuracy_weighted_neutral(signals: dict, accuracies: dict):
    weighted_sum = 0
    total_weight = 0
    for name, signal in signals.items():
        acc = accuracies.get(name, {"up": 0.5, "down": 0.5, "neutral": 0.5})
        if signal == 1:
            weighted_sum += acc["up"]
            total_weight += acc["up"]
        elif signal == -1:
            weighted_sum += -acc["down"]
            total_weight += acc["down"]
        elif signal == 0:
            total_weight += acc["neutral"]
    if total_weight == 0:
        return 0
    score = weighted_sum / total_weight
    return 1 if score > 0 else -1 if score < 0 else 0

def majority_vote(signals: dict):
    count = {1: 0, -1: 0, 0: 0}
    for s in signals.values():
        count[int(np.sign(s))] += 1
    if count[1] > count[-1]:
        return 1
    elif count[-1] > count[1]:
        return -1
    else:
        return 0

# Accuracy Tracking
signal_accuracy = defaultdict(lambda: {"up": [], "down": [], "neutral": []})
agg_accuracy = []
equal_accuracy = []
majority_accuracy = []
pending_predictions = []

def evaluate_predictions(df, now):
    global pending_predictions
    current_price = df["price"].iloc[-1]
    updated_queue = []

    for pred in pending_predictions:
        if now >= pred["eval_time"]:
            actual_change = (current_price - pred["price"]) / pred["price"]
            truth = 0 if abs(actual_change) < THRESHOLD else int(np.sign(actual_change))

            for name, value in pred["signals"].items():
                signal = int(np.sign(value))
                if signal == 1:
                    signal_accuracy[name]["up"].append(truth == 1)
                elif signal == -1:
                    signal_accuracy[name]["down"].append(truth == -1)
                elif signal == 0:
                    signal_accuracy[name]["neutral"].append(truth == 0)

            acc = {
                k: {
                    "up": np.mean(v["up"]) if v["up"] else 0.5,
                    "down": np.mean(v["down"]) if v["down"] else 0.5,
                    "neutral": np.mean(v["neutral"]) if v["neutral"] else 0.5,
                }
                for k, v in signal_accuracy.items()
            }

            acc_agg = accuracy_weighted_neutral(pred["signals"], acc)
            acc_equal = equal_weighted(pred["signals"])
            acc_majority = majority_vote(pred["signals"])
            agg_accuracy.append(acc_agg == truth)
            equal_accuracy.append(acc_equal == truth)
            majority_accuracy.append(acc_majority == truth)
        else:
            updated_queue.append(pred)

    pending_predictions = updated_queue

# Main loop
live = LiveTrade()
last_cycle = time.monotonic()

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w") as f:
        f.write("timestamp,price," + ",".join(param_df.index) + ",equal_agg,acc_agg,majority_agg\n")

while True:
    live.get_last_trade()
    df = live.get_dataframe()
    now = time.monotonic()
    evaluate_predictions(df, now)

    if now - last_cycle >= INTERVAL:
        signals = compute_signals(df)
        if not signals:
            time.sleep(INTERVAL)
            continue

        price_now = df["price"].iloc[-1]
        pred_signals = {k: int(np.sign(v)) for k, v in signals.items()}

        equal_agg_signal = equal_weighted(signals)
        majority_agg_signal = majority_vote(signals)

        acc_ready = len(agg_accuracy) > 0 and len(equal_accuracy) > 0
        acc = {
            k: {
                "up": np.mean(v["up"]) if v["up"] else 0.5,
                "down": np.mean(v["down"]) if v["down"] else 0.5,
                "neutral": np.mean(v["neutral"]) if v["neutral"] else 0.5,
            }
            for k, v in signal_accuracy.items()
        } if acc_ready else {}

        acc_agg_signal = accuracy_weighted_neutral(signals, acc) if acc_ready else 0

        pending_predictions.append({
            "timestamp": now,
            "eval_time": now + HORIZON,
            "price": price_now,
            "signals": signals,
            "acc_agg": acc_agg_signal,
            "equal_agg": equal_agg_signal,
            "majority_agg": majority_agg_signal
        })

        acc_agg_pct = f"{np.mean(agg_accuracy) * 100:.1f}%" if acc_ready else "N/A"
        equal_agg_pct = f"{np.mean(equal_accuracy) * 100:.1f}%" if acc_ready else "N/A"
        majority_agg_pct = f"{np.mean(majority_accuracy) * 100:.1f}%" if acc_ready else "N/A"

        summary = ", ".join(
            f"{k}: {int(np.sign(v)):>+d} ({np.mean(signal_accuracy[k]['up'] + signal_accuracy[k]['down'] + signal_accuracy[k]['neutral'])*100:.1f}%)"
            if signal_accuracy[k]['up'] or signal_accuracy[k]['down'] or signal_accuracy[k]['neutral'] else
            f"{k}: {int(np.sign(v)):>+d} (N/A)"
            for k, v in signals.items()
        )

        logging.info(
            f"Price: {price_now:.2f} | T+{HORIZON} pred: {summary} | "
            f"agg_equal: {equal_agg_signal:+d} ({equal_agg_pct}), "
            f"agg_accuracy: {acc_agg_signal:+d} ({acc_agg_pct}), "
            f"agg_majority: {majority_agg_signal:+d} ({majority_agg_pct})"
        )

        row = f"{datetime.utcnow().isoformat()},{price_now}," + ",".join(
            str(pred_signals.get(ind, "")) for ind in param_df.index
        ) + f",{equal_agg_signal},{acc_agg_signal},{majority_agg_signal}"
        with open(CSV_FILE, "a") as f:
            f.write(row + "\n")

        last_cycle = now

    time.sleep(INTERVAL)
