import pandas as pd
import numpy as np
import time
import logging
from signal_class import SignalGenerator  # allows compute_signal dispatch

logging.basicConfig(format='%(asctime)s [%(levelname)-5s]  %(message)s', level=logging.INFO)

# ─── LOAD SELECTED SIGNALS FROM CSV ────────────────────────────────────────────
# CSV should have column 'signal' listing each chosen strategy (e.g. SMA, RSI)
selected_df = pd.read_csv('result/selected_signal_after_wrc_rf.csv')
selected_df.rename(columns={'Unnamed: 0': 'signal'}, inplace=True)
selected_signals = selected_df['signal'].tolist()

# ─── LOAD PARAMS FROM TRAIN-TEST SUMMARY ───────────────────────────────────────
# Expects columns: strategy, short_window, long_window, lookback, std_dev
span_df = pd.read_csv('result/best_param_train_test_summary.csv')
span_df.set_index('strategy', inplace=True)

# ─── INITIALIZE SCORES FOR SELECTED SIGNALS ─────────────────────────────────────
scores = {sig: 1 for sig in selected_signals}
total_combined = 0
correct_combined = 0

# HELPER TO EXTRACT LAST VALUE (handles numpy arrays and pandas Series)
def get_last(sig):
    arr = np.asarray(sig)
    return int(arr[-1])

# ─── DETERMINE MAX ROLLING WINDOW REQUIRED ────────────────────────────────────
# Generic: take max of available window params per signal +1
def compute_rolling_max():
    windows = []
    for sig in selected_signals:
        if sig not in span_df.index:
            raise KeyError(f"Parameters for signal '{sig}' not found in summary CSV.")
        row = span_df.loc[sig]
        vals = []
        for col in ['short_window', 'long_window', 'lookback']:
            v = row.get(col, np.nan)
            if pd.notna(v):
                vals.append(int(v))
        if not vals:
            raise ValueError(f"No valid window size for signal '{sig}'.")
        windows.append(max(vals) + 1)
    return max(windows)

ROLLING_MAX = compute_rolling_max()
CHECK_INTERVAL = 1  # seconds
data_file = 'result/live_data_agg.csv'

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    while True:
        # read most recent data
        data = pd.read_csv(data_file)
        if len(data) < ROLLING_MAX:
            logging.info(f"Insufficient data: {len(data)}/{ROLLING_MAX}")
            time.sleep(CHECK_INTERVAL)
            continue

        tail = data.tail(ROLLING_MAX).reset_index(drop=True)
        tail.rename(columns={'price': 'last_trade_price'}, inplace=True)

        # generate each signal from history only
        sigs = {}
        for sig in selected_signals:
            params = span_df.loc[sig]
            signal_series = SignalGenerator.compute_signal(
                tail,
                strategy=sig,
                short_window=int(params.get('short_window', np.nan)) if pd.notna(params.get('short_window', np.nan)) else None,
                long_window=int(params.get('long_window', np.nan)) if pd.notna(params.get('long_window', np.nan)) else None,
                lookback=int(params.get('lookback', np.nan)) if pd.notna(params.get('lookback', np.nan)) else None,
                std_dev=float(params.get('std_dev', np.nan)) if pd.notna(params.get('std_dev', np.nan)) else None
            )
            sigs[sig] = get_last(signal_series)

        price_now = tail['last_trade_price'].iloc[-1]
        # wait for next tick
        time.sleep(CHECK_INTERVAL)
        new_data = pd.read_csv(data_file)
        price_next = new_data['price'].iloc[-1]

        # update scores
        for sig, val in sigs.items():
            if (val == 1 and price_next > price_now) or (val == 0 and price_next < price_now):
                scores[sig] += 1

        # normalize into weights
        total = sum(scores.values())
        weights = {sig: scores[sig]/total for sig in selected_signals}

        # compute weighted combined signal
        weighted_sum = sum(weights[sig] * sigs[sig] for sig in selected_signals)
        combined = 1 if weighted_sum >= 0.5 else 0

        # update combined accuracy
        is_combined_correct = (combined == 1 and price_next > price_now) or \
                               (combined == 0 and price_next < price_now)
        total_combined += 1
        if is_combined_correct:
            correct_combined += 1
        combined_acc = 100 * correct_combined / total_combined

        # determine actual market direction
        if price_next > price_now:
            true_dir = 1
        elif price_next < price_now:
            true_dir = -1
        else:
            true_dir = 0

        # was our prediction correct?
        is_correct = (combined == true_dir)
        hit_str = "CORRECT" if is_correct else "WRONG"

        # log
        lbl = lambda x: 'BUY' if x == 1 else 'SELL'
        info = " | ".join(f"{sig}: {lbl(sigs[sig])}({weights[sig]*100:.1f}%)" for sig in selected_signals)
        logging.info(f"{info} ⇒ Combined: {lbl(combined)} | "
                     f"{hit_str} | "
                     f"{price_now:.2f}→{price_next:.2f} | Acc: {combined_acc:.2f}%")

        # loop
