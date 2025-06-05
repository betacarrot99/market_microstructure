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

correct_combined = 0
total_combined = 0

def signal_ewma(df_tail):
    """Return EWMA signal (1 or 0) using the second-last bar."""
    df = df_tail.copy()
    df['short_ewma'] = df['price'].ewm(span=EWMA_SHORT, adjust=False).mean()
    df['long_ewma']  = df['price'].ewm(span=EWMA_LONG, adjust=False).mean()
    sig = int(df.iloc[-2]['short_ewma'] > df.iloc[-2]['long_ewma'])
    return sig

def signal_sma(df_tail):
    """Return SMA signal (1 or 0) using the second-last bar."""
    df = df_tail.copy()
    df['short_sma'] = df['price'].rolling(window=SMA_SHORT).mean()
    df['long_sma']  = df['price'].rolling(window=SMA_LONG).mean()
    sig = int(df.iloc[-2]['short_sma'] > df.iloc[-2]['long_sma'])
    return sig

def signal_mean_reversion(df_tail):
    """Return Mean Reversion signal (1 if price < rolling_mean, else 0) on second-last bar."""
    df = df_tail.copy()
    df['rolling_mean'] = df['price'].rolling(window=MR_MEAN_SPAN).mean()
    sig = int(df.iloc[-2]['price'] < df.iloc[-2]['rolling_mean'])
    return sig

def signal_tsmom(df_tail):
    """Return Time-Series Momentum signal (1 if price > price_{-lookback}, else 0)."""
    df = df_tail.copy()
    df['past_price'] = df['price'].shift(TS_LOOKBACK)
    sig = int(df.iloc[-2]['price'] > df.iloc[-2]['past_price'])
    return sig

def signal_logreg(df_tail):
    """Return Logistic Regression signal (1 or 0) using lag features on second-last bar."""
    df = df_tail.copy()
    # create lag features up to LOGREG_LAG
    for i in range(1, LOGREG_LAG + 1):
        df[f'lag_{i}'] = df['price'].shift(i)
    df = df.dropna().reset_index(drop=True)
    # take the second-last row's lag features as X
    X_live = df[[f'lag_{i}' for i in range(1, LOGREG_LAG + 1)]]
    sig = int(LOGREG_MODEL.predict(X_live.iloc[[-2]])[0])
    return sig

while True:
    try:
        data = pd.read_csv(CSV_FILE)
        if len(data) < ROLLING_WINDOW + 1:
            logging.info(f"Not enough data yet: {len(data)} rows (< {ROLLING_WINDOW+1})")
        else:
            # Take the last ROLLING_WINDOW+1 rows
            tail = data.tail(ROLLING_WINDOW + 1).reset_index(drop=True)

            # Compute each method's signal
            s_ewma = signal_ewma(tail)
            s_sma  = signal_sma(tail)
            s_mr   = signal_mean_reversion(tail)
            s_tsm  = signal_tsmom(tail)
            s_lr   = signal_logreg(tail)

            individual_signals = [s_ewma, s_sma, s_mr, s_tsm, s_lr]
            sum_signals = sum(individual_signals)
            # Combine by majority (>=3 out of 5 → combined=1)
            combined_signal = 1 if sum_signals >= 3 else 0

            # Evaluate combined prediction
            price_now = tail.iloc[-2]['price']
            price_next = tail.iloc[-1]['price']
            is_correct = (
                (combined_signal == 1 and price_next > price_now) or
                (combined_signal == 0 and price_next < price_now)
            )

            if is_correct:
                correct_combined += 1
            total_combined += 1
            combined_accuracy = 100 * correct_combined / total_combined

            # Log all signals and combined result
            logging.info(
                f"EWMA: {s_ewma} | SMA: {s_sma} | MR: {s_mr} | TSMOM: {s_tsm} | LogReg: {s_lr} "
                f"⇒ Combined: {'UP' if combined_signal==1 else 'DOWN'} | "
                f"Price Now: {price_now:.2f} → Next: {price_next:.2f} | "
                f"Result: {'correct' if is_correct else 'wrong'} | "
                f"Accuracy: {combined_accuracy:.2f}%"
            )
    except Exception as e:
        logging.warning(f"Error in main loop: {e}")

    time.sleep(CHECK_INTERVAL)
