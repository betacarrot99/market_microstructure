#!/usr/bin/env python3
"""
4.3.weight_rf_backtest.py

Backtest RF-weighted signal strategy on train/test datasets.
Inputs and outputs mirror weight_equal_backtest_class.py, and it
prints Train, Test, and Overall accuracy.
"""

import pandas as pd
import numpy as np
from signal_class import SignalGenerator  # your centralized signal logic

# ─── LOAD RF IMPORTANCES & SELECTED SIGNALS ────────────────────────────────────
imp_df = pd.read_csv('result/selected_signal_after_wrc_rf.csv', index_col=0)
selected_signals = imp_df.index.tolist()
raw_importances = imp_df['importance'].to_dict()
total_imp = sum(raw_importances[sig] for sig in selected_signals)
weights = {sig: raw_importances[sig] / total_imp for sig in selected_signals}

# ─── LOAD BEST PARAMS ─────────────────────────────────────────────────────────
span_df = (
    pd.read_csv('result/best_param_train_test_summary.csv')
    .set_index('strategy')
)


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def get_last(sig_series):
    arr = np.asarray(sig_series)
    return int(arr[-1])


def compute_rolling_window(signals):
    windows = []
    for sig in signals:
        if sig not in span_df.index:
            raise KeyError(f"Missing params for signal '{sig}'")
        row = span_df.loc[sig]
        vals = [
            int(row[c]) for c in ('short_window', 'long_window', 'lookback')
            if pd.notna(row.get(c))
        ]
        if not vals:
            raise ValueError(f"No windows for '{sig}'")
        windows.append(max(vals) + 1)
    return max(windows)


ROLL_WINDOW = compute_rolling_window(selected_signals)


# ─── BACKTEST FUNCTION ────────────────────────────────────────────────────────
def backtest_accuracy_rf(file_path):
    df = pd.read_csv(file_path)
    # normalize price column
    if 'price' in df.columns:
        df = df.rename(columns={'price': 'last_trade_price'})

    total = 0
    correct = 0

    for i in range(ROLL_WINDOW, len(df)):
        window = df.iloc[i - ROLL_WINDOW:i].reset_index(drop=True)
        price_now = window['last_trade_price'].iloc[-1]
        price_next = df['last_trade_price'].iloc[i]

        # compute each signal value (0/1)
        sigs = {}
        for sig in selected_signals:
            params = span_df.loc[sig]
            series = SignalGenerator.compute_signal(
                window,
                strategy=sig,
                short_window=int(params['short_window']) if pd.notna(params.get('short_window')) else None,
                long_window=int(params['long_window']) if pd.notna(params.get('long_window')) else None,
                lookback=int(params['lookback']) if pd.notna(params.get('lookback')) else None,
                std_dev=float(params['std_dev']) if pd.notna(params.get('std_dev')) else None,
            )
            sigs[sig] = get_last(series)

        # RF-weighted combine → action 1 (buy) or 0 (sell)
        score = sum(weights[sig] * sigs[sig] for sig in selected_signals)
        action = 1 if score >= 0.5 else 0

        # true direction: 1 if up, 0 if down or flat
        true_dir = 1 if price_next > price_now else 0
        if price_next != price_now:
            total += 1
            correct += (action == true_dir)

    accuracy = 100 * correct / total if total > 0 else float('nan')
    return correct, total, accuracy


# ─── MAIN: RUN ON TRAIN & TEST SETS ───────────────────────────────────────────
if __name__ == '__main__':
    train_file = 'result/train_backtesting.csv'
    test_file = 'result/test_backtesting.csv'

    tr_c, tr_t, tr_acc = backtest_accuracy_rf(train_file)
    te_c, te_t, te_acc = backtest_accuracy_rf(test_file)

    ov_c = tr_c + te_c
    ov_t = tr_t + te_t
    ov_acc = 100 * ov_c / ov_t if ov_t > 0 else float('nan')

    print(f"Train Accuracy:    {tr_acc:.2f}% ({tr_c}/{tr_t})")
    print(f"Test Accuracy:     {te_acc:.2f}% ({te_c}/{te_t})")
    print(f"Overall Accuracy:  {ov_acc:.2f}% ({ov_c}/{ov_t})")
