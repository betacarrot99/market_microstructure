import pandas as pd
import numpy as np
import time
import logging
from signal_class import SignalGenerator  # centralized signal logic

logging.basicConfig(format='%(asctime)s [%(levelname)-5s]  %(message)s', level=logging.INFO)

# ─── LOAD SELECTED SIGNALS ─────────────────────────────────────────────────────
selected_df = pd.read_csv('result/selected_signal_after_wrc_rf.csv')
selected_df.rename(columns={'Unnamed: 0': 'signal'}, inplace=True)
selected_signals = selected_df['signal'].tolist()

# ─── LOAD PARAMS ───────────────────────────────────────────────────────────────
span_df = pd.read_csv('result/best_param_train_test_summary.csv')
span_df.set_index('strategy', inplace=True)

# ─── TRACK COMBINED ACCURACY ───────────────────────────────────────────────────
total_combined = 0
correct_combined = 0

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def get_last(sig):
    arr = np.asarray(sig)
    return int(arr[-1])

# Determine max lookback needed for all selected signals
def compute_rolling_max():
    windows = []
    for sig in selected_signals:
        if sig not in span_df.index:
            raise KeyError(f"Missing params for '{sig}'")
        row = span_df.loc[sig]
        vals = []
        for col in ['short_window', 'long_window', 'lookback']:
            v = row.get(col, np.nan)
            if pd.notna(v):
                vals.append(int(v))
        if not vals:
            raise ValueError(f"No window param for '{sig}'")
        windows.append(max(vals) + 1)
    return max(windows)

ROLLING_MAX = compute_rolling_max()
CHECK_INTERVAL = 1  # seconds
DATA_FILE = 'result/live_data_agg.csv'

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Precompute equal weight for fixed-length vector
    weight = 1.0 / len(selected_signals)
    weights = {sig: weight for sig in selected_signals}

    while True:
        df = pd.read_csv(DATA_FILE)
        if len(df) < ROLLING_MAX:
            logging.info(f"Insufficient data: {len(df)}/{ROLLING_MAX}")
            time.sleep(CHECK_INTERVAL)
            continue

        window = df.tail(ROLLING_MAX).reset_index(drop=True)
        window.rename(columns={'price': 'last_trade_price'}, inplace=True)

        # Generate signals purely from history
        sigs = {}
        for sig in selected_signals:
            params = span_df.loc[sig]
            series = SignalGenerator.compute_signal(
                window,
                strategy=sig,
                short_window=int(params.get('short_window', np.nan)) if pd.notna(params.get('short_window', np.nan)) else None,
                long_window=int(params.get('long_window', np.nan)) if pd.notna(params.get('long_window', np.nan)) else None,
                lookback=int(params.get('lookback', np.nan)) if pd.notna(params.get('lookback', np.nan)) else None,
                std_dev=float(params.get('std_dev', np.nan)) if pd.notna(params.get('std_dev', np.nan)) else None
            )
            sigs[sig] = get_last(series)

        price_now = window['last_trade_price'].iloc[-1]
        time.sleep(CHECK_INTERVAL)
        df_new = pd.read_csv(DATA_FILE)
        price_next = df_new['price'].iloc[-1]

        # Weighted combine with equal weights
        weighted_sum = sum(weights[sig] * sigs[sig] for sig in selected_signals)
        combined = 1 if weighted_sum >= 0.5 else 0

        # Update combined accuracy
        is_correct = (combined == 1 and price_next > price_now) or (combined == 0 and price_next < price_now)
        total_combined += 1
        if is_correct:
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

        # Logging
        lbl = lambda x: 'BUY' if x == 1 else 'SELL'
        info = " | ".join(f"{sig}: {lbl(sigs[sig])}({weights[sig]*100:.1f}%)" for sig in selected_signals)
        logging.info(
            f"{info} ⇒ Combined: {lbl(combined)} | "
            f"{hit_str} | "
            f"{price_now:.2f}→{price_next:.2f} | Acc: {combined_acc:.2f}%"
        )

        # Next iteration
