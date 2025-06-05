import logging
import time

import joblib
import pandas as pd
import numpy as np

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

# For live PnL tracking using log returns
prices_list = []
positions_list = []
prev_price_next = None

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
            for name, sig in [
                ('EWMA', sig_ewma),
                ('SMA', sig_sma),
                ('MR', sig_mr),
                ('TSMOM', sig_tsm),
                ('LogReg', sig_lr)
            ]:
                total_counts[name] += 1
                if sig == true_dir:
                    correct_counts[name] += 1

            # Compute raw accuracies (0.0 to 1.0)
            accuracies = {
                name: (correct_counts[name] / total_counts[name])
                for name in correct_counts
            }

            # Normalize so total weight = 1.0
            acc_values = [
                accuracies['EWMA'],
                accuracies['SMA'],
                accuracies['MR'],
                accuracies['TSMOM'],
                accuracies['LogReg']
            ]
            sum_acc = sum(acc_values)
            if sum_acc == 0:
                # If all accuracies are zero (initial case), assign equal weights
                weights = [1/5] * 5
            else:
                weights = [acc / sum_acc for acc in acc_values]

            # Weighted sum of signals using normalized weights
            weighted_sum = (
                weights[0] * sig_ewma +
                weights[1] * sig_sma +
                weights[2] * sig_mr +
                weights[3] * sig_tsm +
                weights[4] * sig_lr
            )

            # Combined signal: +1 if weighted_sum >= 0, else -1
            combined_signal = 1 if weighted_sum >= 0 else -1

            # Only append if this bar is new
            if prev_price_next is None or price_next != prev_price_next:
                prices_list.append(price_next)
                positions_list.append(combined_signal)
                prev_price_next = price_next

            # Compute cumulative returns when we have at least two prices
            if len(prices_list) >= 2:
                prices_arr = np.array(prices_list)
                positions_arr = np.array(positions_list)

                # Log returns for market (SPY_return)
                spy_returns = np.log(prices_arr[1:] / prices_arr[:-1])

                # Strategy return = position_{t-1} * spy_return_t
                strat_returns = positions_arr[:-1] * spy_returns

                cum_market_return = spy_returns.cumsum()[-1]
                cum_strategy_return = strat_returns.cumsum()[-1]
            else:
                cum_market_return = 0.0
                cum_strategy_return = 0.0

            # Update combined accuracy
            total_combined += 1
            if combined_signal == true_dir:
                correct_combined += 1
            combined_accuracy = correct_combined / total_combined

            # Log everything
            logging.info(
                f"Signals → EWMA: {sig_ewma}, SMA: {sig_sma}, MR: {sig_mr}, "
                f"TSMOM: {sig_tsm}, LogReg: {sig_lr} | "
                f"Weights → EWMA: {weights[0]:.2f}, "
                f"SMA: {weights[1]:.2f}, MR: {weights[2]:.2f}, "
                f"TSMOM: {weights[3]:.2f}, "
                f"LogReg: {weights[4]:.2f} | "
                f"Weighted Sum: {weighted_sum:.2f} | "
                f"Combined: {'UP' if combined_signal == 1 else 'DOWN'} | "
                f"Price Now: {price_now:.2f} → Next: {price_next:.2f} | "
                f"Cum Market Return: {cum_market_return:.4%} | "
                f"Cum Strat Return: {cum_strategy_return:.4%} | "
                f"Combined Acc: {combined_accuracy:.2%}"
            )
    except Exception as e:
        logging.warning(f"Error in main loop: {e}")

    time.sleep(CHECK_INTERVAL)
