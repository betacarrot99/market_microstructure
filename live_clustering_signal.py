import os
import time
import logging
import pandas as pd
import numpy as np
import requests
from collections import defaultdict
from datetime import datetime, timezone, timedelta # Add timezone
import joblib 
import json   

# Import the class, not individual functions from it
from trading_indicators import TradingIndicators

logging.basicConfig(format='%(asctime)s [%(levelname)-5s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Config
TRADE_URL = 'https://fapi.binance.com/fapi/v1/trades'
SYMBOL = "BTCUSDT"
# INTERVAL: How often to generate a new prediction.
# Needs to be > average time to fetch data + compute features + predict.
INTERVAL = 1      # Prediction interval in seconds (adjust based on computation time)
# HORIZON: How far into the future the prediction is for.
HORIZON = 20       # Evaluation horizon in seconds (e.g., predict price change in next 10s)
# MAXLEN: Size of the data buffer for features. Should be > largest lookback_window from config.
MAXLEN = 500      # Data buffer size.
CSV_FILE = "live_clustering_predictions_v2.csv"
THRESHOLD = 0.0001  # 0.01% threshold for significant price change (up/down/neutral)

MODEL_ARTIFACTS_DIR = "model_artifacts"
KMEANS_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "best_kmeans_model.joblib")
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "best_scaler.joblib")
CLUSTERING_CONFIG_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "best_clustering_config.json")

KMEANS_MODEL = None
SCALER = None
CLUSTERING_CONFIG = None
ACTIVE_SIGNALS_LIST = [] 

def load_model_and_config():
    global KMEANS_MODEL, SCALER, CLUSTERING_CONFIG, ACTIVE_SIGNALS_LIST, MAXLEN
    try:
        if not os.path.exists(KMEANS_MODEL_PATH):
            logging.error(f"KMeans model file not found: {KMEANS_MODEL_PATH}")
            return False
        if not os.path.exists(SCALER_PATH):
            logging.error(f"Scaler file not found: {SCALER_PATH}")
            return False
        if not os.path.exists(CLUSTERING_CONFIG_PATH):
            logging.error(f"Clustering config file not found: {CLUSTERING_CONFIG_PATH}")
            return False
            
        KMEANS_MODEL = joblib.load(KMEANS_MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        with open(CLUSTERING_CONFIG_PATH, 'r') as f:
            CLUSTERING_CONFIG = json.load(f)
        
        logging.info("Clustering model, scaler, and config loaded successfully.")
        logging.info(f"Config: Lookback={CLUSTERING_CONFIG.get('lookback_window')}, N_Clusters={CLUSTERING_CONFIG.get('n_clusters')}, Features={CLUSTERING_CONFIG.get('features')}")
        
        # Adjust MAXLEN if necessary based on loaded lookback_window
        loaded_lookback = CLUSTERING_CONFIG.get('lookback_window', 20)
        if MAXLEN <= loaded_lookback + 20: # Ensure MAXLEN is sufficiently larger
            new_maxlen = loaded_lookback + 50 # Example adjustment
            logging.warning(f"Initial MAXLEN ({MAXLEN}) might be too small for lookback {loaded_lookback}. Adjusting to {new_maxlen}.")
            MAXLEN = new_maxlen

        if "Clustering" not in ACTIVE_SIGNALS_LIST: # Avoid duplicates if reloaded
            ACTIVE_SIGNALS_LIST.append("Clustering")
        return True
    except Exception as e:
        logging.error(f"Error loading clustering model/config: {e}. Clustering signal will be disabled.", exc_info=True)
        KMEANS_MODEL = None
        SCALER = None
        CLUSTERING_CONFIG = None
        return False

class LiveTrade:
    def __init__(self, symbol=SYMBOL, maxlen=MAXLEN): # maxlen will be updated if CLUSTERING_CONFIG changes it
        self.symbol = symbol
        self._maxlen = maxlen # Use internal _maxlen
        self.prices, self.timestamps, self.volumes = [], [], []

    @property
    def maxlen(self):
        return self._maxlen

    @maxlen.setter
    def maxlen(self, value):
        self._maxlen = value
        # Trim existing data if maxlen is reduced
        self.prices = self.prices[-self._maxlen:]
        self.timestamps = self.timestamps[-self._maxlen:]
        self.volumes = self.volumes[-self._maxlen:]
        logging.info(f"LiveTrade maxlen updated to {self._maxlen}")


    def get_last_trade(self):
        try:
            response = requests.get(TRADE_URL, params={'symbol': self.symbol, 'limit': 1}, timeout=3)
            response.raise_for_status()
            trade = response.json()[0]
            
            if not self.timestamps or trade['time'] > self.timestamps[-1]:
                self.timestamps.append(trade['time'])
                self.prices.append(float(trade['price']))
                self.volumes.append(float(trade['qty']))

                self.timestamps = self.timestamps[-self._maxlen:]
                self.prices = self.prices[-self._maxlen:]
                self.volumes = self.volumes[-self._maxlen:]
        except requests.exceptions.RequestException as e:
            logging.warning(f"Trade fetch HTTP error: {e}")
        except Exception as e:
            logging.error(f"Trade fetch generic error: {e}", exc_info=False) # Set exc_info=False for less verbose logs on common errors

    def get_dataframe(self):
        return pd.DataFrame({
            "time": pd.to_datetime(self.timestamps, unit='ms'),
            "price": self.prices,
            "volume": self.volumes
        })

def compute_live_clustering_signal(df_live_trades_input):
    if not KMEANS_MODEL or not SCALER or not CLUSTERING_CONFIG:
        logging.debug("Clustering model/scaler/config not loaded. Skipping signal.")
        return 0 

    lookback = CLUSTERING_CONFIG['lookback_window']
    feature_list = CLUSTERING_CONFIG['features']
    n_clusters_cfg = CLUSTERING_CONFIG['n_clusters']

    if 'price' not in df_live_trades_input.columns:
        logging.error("Input DataFrame for clustering signal missing 'price' column.")
        return 0
    
    df_for_features = df_live_trades_input.rename(columns={'price': 'last_trade_price'})

    min_rows_for_features = lookback # Need at least 'lookback' rows to get one non-NaN feature set
    if len(df_for_features) < min_rows_for_features:
        logging.debug(f"Not enough data rows ({len(df_for_features)}) for feature calculation (need {min_rows_for_features}).")
        return 0

    try:
        df_with_tech_features = TradingIndicators.create_technical_features(df_for_features, lookback)
    except Exception as e:
        logging.error(f"Error in create_technical_features: {e}", exc_info=True)
        return 0

    feature_data_unscaled = df_with_tech_features[feature_list].dropna()

    if feature_data_unscaled.empty:
        logging.debug("No valid (non-NaN) feature data rows after dropna().")
        return 0
    
    min_rows_for_signal_logic = 2 # generate_cluster_signals needs at least 2 rows for prev_cluster logic
    if len(feature_data_unscaled) < min_rows_for_signal_logic:
        logging.debug(f"Too few feature data points ({len(feature_data_unscaled)}) after dropna for signal logic (need {min_rows_for_signal_logic}).")
        return 0

    try:
        features_scaled = SCALER.transform(feature_data_unscaled)
    except ValueError as e:
        logging.error(f"Error scaling features: {e}. Scaler expected {SCALER.n_features_in_} features, got {feature_data_unscaled.shape[1]} from features: {feature_list}", exc_info=True)
        return 0
    except Exception as e:
        logging.error(f"Unexpected error scaling features: {e}", exc_info=True)
        return 0

    try:
        clusters_pred = KMEANS_MODEL.predict(features_scaled)
    except Exception as e:
        logging.error(f"Error predicting clusters: {e}", exc_info=True)
        return 0

    df_for_signal_generation = df_with_tech_features.loc[feature_data_unscaled.index].copy()
    df_for_signal_generation['cluster'] = clusters_pred
    
    try:
        df_with_final_signals = TradingIndicators.generate_cluster_signals(df_for_signal_generation, n_clusters_cfg)
    except Exception as e:
        logging.error(f"Error in generate_cluster_signals: {e}", exc_info=True)
        return 0

    if df_with_final_signals.empty or 'Signal' not in df_with_final_signals.columns:
        logging.warning("Clustering signal generation resulted in empty/incomplete DataFrame.")
        return 0
        
    latest_raw_signal = df_with_final_signals['Signal'].iloc[-1]
    return 1 if latest_raw_signal == 1 else -1


def compute_signals(df_live): 
    signals = {}
    if "Clustering" in ACTIVE_SIGNALS_LIST and CLUSTERING_CONFIG:
        try:
            cluster_sig_val = compute_live_clustering_signal(df_live.copy())
            signals["Clustering"] = cluster_sig_val
        except Exception as e:
            logging.error("Unhandled error in compute_live_clustering_signal wrapper:", exc_info=True)
            signals["Clustering"] = 0 
    return signals

signal_accuracy_stats = defaultdict(lambda: {"total_pred": 0, "correct_pred": 0, "up_pred":0, "up_correct":0, "down_pred":0, "down_correct":0, "neutral_pred":0, "neutral_correct":0})
agg_accuracy_records = { "EqualWeighted": [], "MajorityVote": [] }
pending_predictions = []

def evaluate_predictions(current_price_eval, current_timestamp_eval_dt):
    global pending_predictions
    updated_queue = []

    for pred_item in pending_predictions:
        if current_timestamp_eval_dt >= pred_item["eval_time_dt"]:
            price_at_pred = pred_item["price_at_pred"]
            actual_change_pct = (current_price_eval - price_at_pred) / price_at_pred
            
            truth = 1 if actual_change_pct > THRESHOLD else -1 if actual_change_pct < -THRESHOLD else 0

            for sig_name, sig_value_predicted in pred_item["signals"].items():
                stats = signal_accuracy_stats[sig_name]
                stats["total_pred"] += 1
                if sig_value_predicted == truth: stats["correct_pred"] += 1
                if sig_value_predicted == 1:
                    stats["up_pred"] +=1
                    if truth == 1: stats["up_correct"] +=1
                elif sig_value_predicted == -1:
                    stats["down_pred"] +=1
                    if truth == -1: stats["down_correct"] +=1
                elif sig_value_predicted == 0: # Should not happen with current clustering signal map
                    stats["neutral_pred"] +=1
                    if truth == 0: stats["neutral_correct"] +=1
            
            # Evaluate aggregated signals
            agg_accuracy_records["EqualWeighted"].append(pred_item["equal_weighted_signal"] == truth)
            agg_accuracy_records["MajorityVote"].append(pred_item["majority_vote_signal"] == truth)
        else: 
            updated_queue.append(pred_item)
    pending_predictions = updated_queue

def get_accuracy_percentage(correct, total):
    return (correct / total * 100) if total > 0 else 0.0 # Return float

def equal_weighted_signal(signals_dict: dict):
    if not signals_dict: return 0
    net_signal = sum(np.sign(s) for s in signals_dict.values() if s is not None) # handle None signals
    return int(np.sign(net_signal))

def majority_vote_signal(signals_dict: dict):
    if not signals_dict: return 0
    votes = defaultdict(int)
    for s_val in signals_dict.values():
        if s_val is not None: # handle None signals
            votes[int(np.sign(s_val))] += 1
    
    if votes[1] > votes[-1] and votes[1] >= votes[0]: return 1
    if votes[-1] > votes[1] and votes[-1] >= votes[0]: return -1
    return 0

if __name__ == "__main__":
    if not load_model_and_config():
        logging.critical("Failed to load model/config. Exiting.")
        exit()
    
    live_data_fetcher = LiveTrade(symbol=SYMBOL, maxlen=MAXLEN) # maxlen might be updated by load_model_and_config
    live_data_fetcher.maxlen = MAXLEN # Ensure LiveTrade instance uses potentially updated MAXLEN

    if not os.path.exists(CSV_FILE):
        header_cols = ["timestamp_utc", "price_at_pred"] + ACTIVE_SIGNALS_LIST + ["equal_weighted", "majority_vote", "eval_price", "actual_direction"]
        with open(CSV_FILE, "w") as f:
            f.write(",".join(header_cols) + "\n")

    last_prediction_time = time.monotonic()

    logging.info(f"Starting live signal generation for {SYMBOL}. Interval: {INTERVAL}s, Horizon: {HORIZON}s, Buffer: {live_data_fetcher.maxlen} trades.")

    while True:
        current_loop_time = time.monotonic()
        live_data_fetcher.get_last_trade()
        
        if not live_data_fetcher.prices:
            time.sleep(0.2) 
            continue

        current_price = live_data_fetcher.prices[-1]
        current_timestamp_ms = live_data_fetcher.timestamps[-1]
        current_datetime_utc = datetime.fromtimestamp(current_timestamp_ms / 1000.0, tz=timezone.utc)

        evaluate_predictions(current_price, current_datetime_utc)

        if current_loop_time - last_prediction_time >= INTERVAL:
            df_current_buffer = live_data_fetcher.get_dataframe()
            
            # Basic check for buffer size related to lookback. More detailed checks are in compute_live_clustering_signal
            required_buffer_for_pred = CLUSTERING_CONFIG.get('lookback_window', 20) + 5 
            if df_current_buffer.empty or len(df_current_buffer) < required_buffer_for_pred:
                logging.debug(f"Buffer size {len(df_current_buffer)} too small for prediction (need ~{required_buffer_for_pred}). Waiting for more data.")
                # Don't reset last_prediction_time here, allow it to retry soon.
                time.sleep(0.2) # Short sleep before next attempt
                continue

            all_signals_now = compute_signals(df_current_buffer)

            if not all_signals_now: # If compute_signals returns empty (e.g. clustering disabled and no other signals)
                logging.debug("No signals computed in this cycle.")
                last_prediction_time = current_loop_time # Important to advance timer
                time.sleep(max(0.05, INTERVAL / 10))
                continue

            eq_w_agg_sig = equal_weighted_signal(all_signals_now)
            maj_v_agg_sig = majority_vote_signal(all_signals_now)

            eval_time_dt = current_datetime_utc + pd.Timedelta(seconds=HORIZON)
            pending_predictions.append({
                "pred_time_dt": current_datetime_utc,
                "eval_time_dt": eval_time_dt,
                "price_at_pred": current_price,
                "signals": all_signals_now.copy(),
                "equal_weighted_signal": eq_w_agg_sig,
                "majority_vote_signal": maj_v_agg_sig,
            })
            
            log_msg_parts = [f"Price: {current_price:.2f}"]
            for sig_name in ACTIVE_SIGNALS_LIST: # Only log active signals
                sig_val = all_signals_now.get(sig_name, 0)
                stats = signal_accuracy_stats[sig_name]
                acc_pct = get_accuracy_percentage(stats['correct_pred'], stats['total_pred'])
                log_msg_parts.append(f"{sig_name}: {sig_val:+d} (Acc: {acc_pct:.1f}%)")
            
            eq_w_acc = np.mean(agg_accuracy_records["EqualWeighted"]) * 100 if agg_accuracy_records["EqualWeighted"] else 0.0
            maj_v_acc = np.mean(agg_accuracy_records["MajorityVote"]) * 100 if agg_accuracy_records["MajorityVote"] else 0.0

            log_msg_parts.append(f"EQ_W: {eq_w_agg_sig:+d} (Acc: {eq_w_acc:.1f}%)")
            log_msg_parts.append(f"MAJ_V: {maj_v_agg_sig:+d} (Acc: {maj_v_acc:.1f}%)")
            logging.info(" | ".join(log_msg_parts) + f" | Pending: {len(pending_predictions)}")

            # Write to CSV (only predicted signals, actual outcome logged upon evaluation if needed)
            csv_row_values = [current_datetime_utc.isoformat(timespec='milliseconds'), f"{current_price:.4f}"]
            for sig_name_header in ACTIVE_SIGNALS_LIST: 
                csv_row_values.append(str(all_signals_now.get(sig_name_header, "")))
            csv_row_values.extend([str(eq_w_agg_sig), str(maj_v_agg_sig)])
            # To log actuals, you'd need to modify evaluate_predictions to write to CSV or store more info
            # For now, this CSV logs predictions as they are made.
            # Placeholder for eval_price and actual_direction, filled upon evaluation if CSV were updated then.
            csv_row_values.extend(["", ""]) # For eval_price, actual_direction

            with open(CSV_FILE, "a", newline='') as f: # Add newline='' for csv writer
                # Using simple join, consider csv.writer for proper quoting if features can have commas
                f.write(",".join(csv_row_values) + "\n")

            last_prediction_time = current_loop_time
        
        time.sleep(max(0.05, INTERVAL / 20)) # Shorter sleep to be responsive