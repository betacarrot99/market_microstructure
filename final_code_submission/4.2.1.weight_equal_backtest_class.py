import pandas as pd
import numpy as np
import time
import logging
from signal_class import SignalGenerator  # centralized signal logic

logging.basicConfig(format='%(asctime)s [%(levelname)-5s]  %(message)s', level=logging.INFO)

# ─── LOAD SELECTED SIGNALS ─────────────────────────────────────────────────────
selected_df = pd.read_csv('result/selected_signal_after_wrc.csv')
selected_signals = selected_df['signal'].tolist()

# ─── LOAD PARAMS ───────────────────────────────────────────────────────────────
span_df = pd.read_csv('result/best_param_train_test_summary.csv')
span_df.set_index('strategy', inplace=True)

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
        vals = [int(row[c]) for c in ['short_window', 'long_window', 'lookback'] if pd.notna(row[c])]
        windows.append(max(vals) + 1)
    return max(windows)

ROLLING_MAX = compute_rolling_max()
# equal weight for each signal
WEIGHT = 1.0 / len(selected_signals)

# ─── STATIC BACKTEST FUNCTION ──────────────────────────────────────────────────
def backtest_accuracy(file_path):
    df = pd.read_csv(file_path)
    # normalize column name
    if 'price' in df.columns:
        df.rename(columns={'price': 'last_trade_price'}, inplace=True)

    total = 0
    correct = 0

    for i in range(ROLLING_MAX, len(df)):
        window = df.iloc[i-ROLLING_MAX:i].reset_index(drop=True)
        price_now  = window['last_trade_price'].iloc[-1]
        price_next = df['last_trade_price'].iloc[i]

        # compute individual signals
        sigs = {}
        for sig in selected_signals:
            params = span_df.loc[sig]
            series = SignalGenerator.compute_signal(
                window,
                strategy     = sig,
                short_window = int(params['short_window']) if pd.notna(params['short_window']) else None,
                long_window  = int(params['long_window'])  if pd.notna(params['long_window'])  else None,
                lookback     = int(params['lookback'])     if pd.notna(params['lookback'])     else None,
                std_dev      = float(params['std_dev'])    if pd.notna(params['std_dev'])      else None
            )
            sigs[sig] = get_last(series)

        # equal-weighted combine
        weighted_sum = sum(WEIGHT * val for val in sigs.values())
        combined = 1 if weighted_sum >= 0.5 else 0

        # true direction: 1 if up, 0 if down or flat
        true_dir = 1 if price_next > price_now else 0
        if price_next != price_now:
            total += 1
            correct += (combined == true_dir)

    accuracy = 100 * correct / total if total > 0 else float('nan')
    return correct, total, accuracy

# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # backtest on train and test datasets
    train_file = 'result/train_backtesting.csv'
    test_file  = 'result/test_backtesting.csv'

    train_corr, train_tot, train_acc = backtest_accuracy(train_file)
    test_corr,  test_tot,  test_acc  = backtest_accuracy(test_file)

    overall_corr = train_corr + test_corr
    overall_tot  = train_tot  + test_tot
    overall_acc  = 100 * overall_corr / overall_tot if overall_tot > 0 else float('nan')

    print(f"Train Accuracy:    {train_acc:.2f}% ({train_corr}/{train_tot})")
    print(f"Test Accuracy:     {test_acc:.2f}% ({test_corr}/{test_tot})")
    print(f"Overall Accuracy:  {overall_acc:.2f}% ({overall_corr}/{overall_tot})")

    # live loop continues below if desired
    while True:
        df = pd.read_csv('result/live_data_agg.csv')
        if len(df) < ROLLING_MAX:
            logging.info(f"Insufficient data: {len(df)}/{ROLLING_MAX}")
            time.sleep(1)
            continue

        window = df.tail(ROLLING_MAX).reset_index(drop=True)
        window.rename(columns={'price': 'last_trade_price'}, inplace=True)

        sigs = {}
        for sig in selected_signals:
            params = span_df.loc[sig]
            series = SignalGenerator.compute_signal(
                window,
                strategy     = sig,
                short_window = int(params['short_window']) if pd.notna(params['short_window']) else None,
                long_window  = int(params['long_window'])  if pd.notna(params['long_window'])  else None,
                lookback     = int(params['lookback'])     if pd.notna(params['lookback'])     else None,
                std_dev      = float(params['std_dev'])    if pd.notna(params['std_dev'])      else None
            )
            sigs[sig] = get_last(series)

        price_now = window['last_trade_price'].iloc[-1]
        time.sleep(1)
        df_new = pd.read_csv('result/live_data_agg.csv')
        price_next = df_new['price'].iloc[-1]

        weighted_sum = sum(WEIGHT * val for val in sigs.values())
        combined = 1 if weighted_sum >= 0.5 else 0

        is_correct = (combined == 1 and price_next > price_now) or (combined == 0 and price_next < price_now)
        hit_str = 'CORRECT' if is_correct else 'WRONG'

        lbl = lambda x: 'BUY' if x == 1 else 'SELL'
        # update weight display using WEIGHT
        info = " | ".join(f"{sig}:{lbl(sigs[sig])}({WEIGHT*100:.1f}%)" for sig in selected_signals)
        logging.info(
            f"{info} ⇒ Combined: {lbl(combined)} | {hit_str} | {price_now:.2f}→{price_next:.2f} | Acc: {(100 if is_correct else 0):.2f}%"
        )
