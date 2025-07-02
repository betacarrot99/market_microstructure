#!/usr/bin/env python3
"""
Live RF-weighted signal execution using dynamic imports from SignalGenerator.
Only uses signals and their importances as specified in the selected_signal_after_wrc_rf.csv.
"""
import os
import csv
import time
import logging
from datetime import datetime

import pandas as pd
import numpy as np

from signal_class import SignalGenerator  # centralized signal logic

# ─── SETUP LOGGING ─────────────────────────────────────────────────────────────
logging.basicConfig(format='%(asctime)s [%(levelname)-5s] %(message)s', level=logging.INFO)

# ─── LOAD RF IMPORTANCES & SELECTED SIGNALS ─────────────────────────────────────
# CSV has no header for signal column, use index
imp_df = pd.read_csv('result/selected_signal_after_wrc_rf.csv', index_col=0)
selected_signals = imp_df.index.tolist()
raw_importances = imp_df['importance'].to_dict()
TRANSACTION_COST = 0.1/10000 #according to BINANCE is 10bps, but we're using 0.1bps
# normalize
total_imp = sum(raw_importances[sig] for sig in selected_signals)
weights = {sig: raw_importances[sig] / total_imp for sig in selected_signals}
print(weights)

# ─── LOAD STRATEGY PARAMETERS ──────────────────────────────────────────────────
# Expects columns: strategy, short_window, long_window, lookback, std_dev
span_df = pd.read_csv('result/best_param_train_test_summary.csv').set_index('strategy')

# ─── CALCULATE ROLLING WINDOW REQUIRED ──────────────────────────────────────────
def compute_rolling_window(signals):
    windows = []
    for sig in signals:
        if sig not in span_df.index:
            raise KeyError(f"Parameters for '{sig}' not found.")
        row = span_df.loc[sig]
        vals = []
        for col in ['short_window', 'long_window', 'lookback']:
            v = row.get(col, np.nan)
            if pd.notna(v):
                vals.append(int(v))
        if not vals:
            raise ValueError(f"No window parameter for '{sig}'.")
        windows.append(max(vals) + 1)
    return max(windows)

ROLL_WINDOW = compute_rolling_window(selected_signals)
CSV_FILE = 'result/live_data_agg.csv'
LOG_CSV = 'result/live_pnl_log_RANDOM_FOREST.csv'
CHECK_INTERVAL = 1  # seconds

# ─── INITIALIZE CSV OUTPUT ──────────────────────────────────────────────────────
if not os.path.exists(LOG_CSV):
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    with open(LOG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'price', 'position', 'realized_pnl',
            'unrealized_pnl', 'pnl_total', 'market_pnl', 'accuracy'
        ])

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def get_last(sig_series):
    arr = np.asarray(sig_series)
    return int(arr[-1])

# ─── STATE VARIABLES ───────────────────────────────────────────────────────────
current_position = 0
average_open_price = 0.0
pnl_realized = 0.0
cumulative_market_pnl = 0.0
correct_count = 0
total_count = 1

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.info("Starting dynamic RF-weighted live trading...")
    while True:
        try:
            data = pd.read_csv(CSV_FILE)
            if len(data) < ROLL_WINDOW + 1:
                logging.info(f"Insufficient data: {len(data)}/{ROLL_WINDOW+1}")
                time.sleep(CHECK_INTERVAL)
                continue

            # prepare history (t) and next price (t+1)
            tail = data.tail(ROLL_WINDOW + 1).reset_index(drop=True)
            history = tail.iloc[:-1].rename(columns={'price': 'last_trade_price'})
            price_now = tail.iloc[-2]['price']
            price_next = tail.iloc[-1]['price']

            price_return = price_next - price_now


            # compute each selected signal
            sigs = {}
            for sig in selected_signals:
                params = span_df.loc[sig]
                series = SignalGenerator.compute_signal(
                    history,
                    strategy=sig,
                    short_window=int(params['short_window']) if pd.notna(params.get('short_window')) else None,
                    long_window=int(params['long_window']) if pd.notna(params.get('long_window')) else None,
                    lookback=int(params['lookback']) if pd.notna(params.get('lookback')) else None,
                    std_dev=float(params['std_dev']) if pd.notna(params.get('std_dev')) else None
                )
                sigs[sig] = get_last(series)

            # compute weighted decision
            score = sum(weights[sig] * sigs[sig] for sig in selected_signals)
            action = 1 if score >= 0.5 else -1

            logging.debug(f"Weighted score: {score:.4f}")
            score_history = []
            score_history.append(score)
            if len(score_history) % 500 == 0:  # every 500 steps
                import matplotlib.pyplot as plt

                plt.hist(score_history, bins=50)
                plt.title("Distribution of Weighted Scores")
                plt.xlabel("Score")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.savefig("result/score_histogram.png")
                plt.close()

            # update accuracy
            if price_next != price_now:
                total_count += 1
            correct = (action == 1 and price_next > price_now) or (action == -1 and price_next < price_now)
            correct_count += int(correct)
            accuracy = correct_count / total_count * 100

            # determine actual market direction
            if price_next > price_now:
                true_dir = 1
            elif price_next < price_now:
                true_dir = -1
            else:
                true_dir = 0

            # was our prediction correct?
            is_correct = (action == true_dir)
            # hit_str = "CORRECT" if is_correct else "WRONG"
            if price_next == price_now:
                hit_str = "NO MOVEMENT"
            elif is_correct:
                hit_str = "CORRECT"
            else:
                hit_str = "WRONG"

            # determine desired position and trade qty
            desired_position = max(-1, min(1, current_position + action))
            qty = desired_position - current_position
            # trade-based PnL accounting
            if qty != 0:
                # open/increase
                if current_position == 0 or current_position * qty > 0:
                    total_qty = abs(current_position) + abs(qty)
                    average_open_price = (average_open_price * abs(current_position) + price_now * abs(qty)) / total_qty
                    if price_next != price_now:
                        pnl_realized -= (TRANSACTION_COST * price_now) #ENTRY COST
                # reduce/close/reverse
                else:
                    pnl_realized += (price_now - average_open_price) * current_position - (TRANSACTION_COST * price_now)
                    if abs(qty) > abs(current_position):
                        average_open_price = price_now
                current_position = desired_position

            # unrealized PnL
            pnl_unrealized = (price_next - average_open_price) * current_position
            pnl_total = pnl_realized + pnl_unrealized

            # market PnL
            cumulative_market_pnl += price_return
            # append to CSV
            with open(LOG_CSV, 'a', newline='') as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(), price_now, current_position,
                    pnl_realized, pnl_unrealized, pnl_total,
                    cumulative_market_pnl, accuracy
                ])

            # console log
            label = lambda v: 'BUY' if v == 1 else 'SELL'
            sig_str = ' | '.join(f"{sig}:{label(sigs[sig])}" for sig in selected_signals)
            logging.info(
                f"{sig_str} => Combined:{label(action)} "
                f"| {hit_str}"
                f"| Price:{price_now:.2f}->{price_next:.2f} "
                f"| Acc:{accuracy:.2f}% "
                f"| Realized:{pnl_realized:.2f} "
                f"| Unrealized:{pnl_unrealized:.2f} "
                f"| Market:{cumulative_market_pnl:.2f} "
                

            )

        except Exception as e:
            logging.warning(f"Error in main loop: {e}")
        time.sleep(CHECK_INTERVAL)
