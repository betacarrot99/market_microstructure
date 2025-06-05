import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import joblib

logging.basicConfig(format='%(asctime)s [%(levelname)-5s]  %(message)s', level=logging.INFO)

# ─── LOAD PARAMS FROM sma_param_log3.csv ───────────────────────────────────────
span_df = pd.read_csv("result/sma_param_log3.csv")

# EWMA params
ewma_rows = span_df[span_df['method'] == 'EWMA'].iloc[-1]
EWMA_SHORT = int(ewma_rows['short_window'])
EWMA_LONG  = int(ewma_rows['long_window'])
ROLL_EWMA  = max(EWMA_SHORT, EWMA_LONG)

# SMA params
sma_rows = span_df[span_df['method'] == 'SMA'].iloc[-1]
SMA_SHORT = int(sma_rows['short_window'])
SMA_LONG  = int(sma_rows['long_window'])
ROLL_SMA  = max(SMA_SHORT, SMA_LONG)

# Mean Reversion params
mr_rows = span_df[span_df['method'] == 'MeanReversion'].iloc[-1]
MR_MEAN_SPAN = int(mr_rows['long_window'])
ROLL_MR      = MR_MEAN_SPAN

# TSMOM params (use short_window as lookback)
tsmom_rows   = span_df[span_df['method'] == 'TSMOM'].iloc[-1]
TS_LOOKBACK  = int(tsmom_rows['short_window'])
ROLL_TSMOM   = TS_LOOKBACK + 1

# LogReg params
logreg_rows  = span_df[span_df['method'] == 'LogReg'].iloc[-1]
LOGREG_LAG   = int(logreg_rows['short_window'])
ROLL_LOGREG  = LOGREG_LAG + 2

# Combined rolling window requirement
ROLLING_WINDOW = max(ROLL_EWMA, ROLL_SMA, ROLL_MR, ROLL_TSMOM, ROLL_LOGREG)
CHECK_INTERVAL = 2  # seconds
CSV_FILE = 'result/live_data.csv'

# Load the trained logistic regression model
LOGREG_MODEL = joblib.load("result/logreg_model.pkl")

# Initialize per-method accuracy counters
correct_counts = {
    'EWMA': 0,
    'SMA': 0,
    'MR': 0,
    'TSMOM': 0,
    'LogReg': 0
}
total_counts = {
    'EWMA': 0,
    'SMA': 0,
    'MR': 0,
    'TSMOM': 0,
    'LogReg': 0
}

# Combined accuracy counters
correct_combined = 0
total_combined = 0

def signal_ewma(df_tail):
    """Return EWMA signal: +1 if short_ewma > long_ewma, else -1."""
    df = df_tail.copy()
    df['short_ewma'] = df['price'].ewm(span=EWMA_SHORT, adjust=False).mean()
    df['long_ewma']  = df['price'].ewm(span=EWMA_LONG, adjust=False).mean()
    return 1 if df.iloc[-2]['short_ewma'] > df.iloc[-2]['long_ewma'] else -1

def signal_sma(df_tail):
    """Return SMA signal: +1 if short_sma > long_sma, else -1."""
    df = df_tail.copy()
    df['short_sma'] = df['price'].rolling(window=SMA_SHORT).mean()
    df['long_sma']  = df['price'].rolling(window=SMA_LONG).mean()
    return 1 if df.iloc[-2]['short_sma'] > df.iloc[-2]['long_sma'] else -1

def signal_mean_reversion(df_tail):
    """Return Mean Reversion signal: +1 if price < rolling_mean, else -1."""
    df = df_tail.copy()
    df['rolling_mean'] = df['price'].rolling(window=MR_MEAN_SPAN).mean()
    return 1 if df.iloc[-2]['price'] < df.iloc[-2]['rolling_mean'] else -1

def signal_tsmom(df_tail):
    """Return Time-Series Momentum signal: +1 if price > past_price, else -1."""
    df = df_tail.copy()
    df['past_price'] = df['price'].shift(TS_LOOKBACK)
    return 1 if df.iloc[-2]['price'] > df.iloc[-2]['past_price'] else -1

def signal_logreg(df_tail):
    """Return Logistic Regression signal: +1 if predict==1, else -1."""
    df = df_tail.copy()
    for i in range(1, LOGREG_LAG + 1):
        df[f'lag_{i}'] = df['price'].shift(i)
    df = df.dropna().reset_index(drop=True)
    if len(df) < 2:
        return -1  # default if insufficient data
    X_live = df[[f'lag_{i}' for i in range(1, LOGREG_LAG + 1)]]
    pred = LOGREG_MODEL.predict(X_live.iloc[[-2]])[0]
    return 1 if pred == 1 else -1

while True:
    try:
        data = pd.read_csv(CSV_FILE)
        if len(data) < ROLLING_WINDOW + 1:
            logging.info(f"Not enough data yet: {len(data)} rows (< {ROLLING_WINDOW + 1})")
        else:
            tail = data.tail(ROLLING_WINDOW + 1).reset_index(drop=True)

            # Compute each method's signal
            sig_ewma = signal_ewma(tail)
            sig_sma  = signal_sma(tail)
            sig_mr   = signal_mean_reversion(tail)
            sig_tsm  = signal_tsmom(tail)
            sig_lr   = signal_logreg(tail)

            # True direction: +1 if price goes up, else -1
            price_now  = tail.iloc[-2]['price']
            price_next = tail.iloc[-1]['price']
            true_dir   = 1 if price_next > price_now else -1

            # Update individual counts & compute accuracies
            for name, sig in [('EWMA', sig_ewma), ('SMA', sig_sma),
                              ('MR', sig_mr), ('TSMOM', sig_tsm),
                              ('LogReg', sig_lr)]:
                total_counts[name] += 1
                if sig == true_dir:
                    correct_counts[name] += 1

            accuracies = {name: (correct_counts[name] / total_counts[name])
                          for name in correct_counts}

            # Weighted sum of signals (weight = accuracy)
            weighted_sum = (
                accuracies['EWMA']  * sig_ewma +
                accuracies['SMA']   * sig_sma +
                accuracies['MR']    * sig_mr +
                accuracies['TSMOM'] * sig_tsm +
                accuracies['LogReg']* sig_lr
            )

            # Combined signal: +1 if weighted_sum >= 0, else -1
            combined_signal = 1 if weighted_sum >= 0 else -1

            # Update combined accuracy
            total_combined += 1
            if combined_signal == true_dir:
                correct_combined += 1
            combined_accuracy = correct_combined / total_combined

            # Log everything
            logging.info(
                f"Signals → EWMA: {sig_ewma}, SMA: {sig_sma}, MR: {sig_mr}, "
                f"TSMOM: {sig_tsm}, LogReg: {sig_lr} | "
                f"Weights → EWMA: {accuracies['EWMA']:.2f}, "
                f"SMA: {accuracies['SMA']:.2f}, MR: {accuracies['MR']:.2f}, "
                f"TSMOM: {accuracies['TSMOM']:.2f}, "
                f"LogReg: {accuracies['LogReg']:.2f} | "
                f"Weighted Sum: {weighted_sum:.2f} | "
                f"Combined: {'UP' if combined_signal == 1 else 'DOWN'} | "
                f"Price Now: {price_now:.2f} → Next: {price_next:.2f} | "
                f"Combined Result: {'correct' if combined_signal == true_dir else 'wrong'} | "
                f"Combined Acc: {combined_accuracy:.2%}"
            )
    except Exception as e:
        logging.warning(f"Error in main loop: {e}")

    time.sleep(CHECK_INTERVAL)
