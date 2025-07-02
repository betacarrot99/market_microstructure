# #!/usr/bin/env python3
# """
# Batch signal aggregation from CSV data, splitting into train/test (80/20).
# Computes raw signals and actual movement only, saving train/test CSVs.
# """
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# # ─── CONFIGURATION ─────────────────────────────────────────────────────────────
# CSV_FILE = 'data/BTCUSDT_1s_resample.csv'
# PARAM_FILE = 'result/best_param_train_test_summary.csv'
# TRAIN_OUT = 'result/signal_agg_train_rf.csv'
# TEST_OUT = 'result/signal_agg_test_rf.csv'
# TEST_SIZE = 0.2
#
# # ─── LOAD BEST PARAMS ──────────────────────────────────────────────────────────
# span_df = pd.read_csv(PARAM_FILE)
# TSMOM_LB = int(span_df.query("strategy=='TSMOM'")['lookback'].iloc[-1])
# RSI_LB = int(span_df.query("strategy=='RSI'")['lookback'].iloc[-1])
# RSI_MOM_LB = int(span_df.query("strategy=='RSI_Momentum'")['lookback'].iloc[-1])
# stoch = span_df.query("strategy=='Stochastic'").iloc[-1]
# STOCH_K, STOCH_D = int(stoch['short_window']), int(stoch['long_window'])
# ATR_LB = int(span_df.query("strategy=='ATR'")['lookback'].iloc[-1])
# ATR_SD = float(span_df.query("strategy=='ATR'")['std_dev'].iloc[-1])
#
# # Rolling window length for signal calculation
# given_signals = ['TSMOM', 'Stochastic', 'RSI', 'RSI_Momentum', 'ATR']
# ROLL_WINDOW = max(TSMOM_LB+1, RSI_LB, RSI_MOM_LB, STOCH_K, STOCH_D, ATR_LB)
#
# # ─── SIGNAL FUNCTIONS ─────────────────────────────────────────────────────────
# def signal_tsmom(df_tail):
#     return float(df_tail['price'].iloc[-1] > df_tail['price'].iloc[-TSMOM_LB-1])
#
# def signal_stochastic(df_tail):
#     low_min = df_tail['low'].rolling(STOCH_K).min()
#     high_max = df_tail['high'].rolling(STOCH_K).max()
#     k = 100 * (df_tail['price'] - low_min) / (high_max - low_min)
#     d = k.rolling(STOCH_D).mean()
#     sig = np.where((k < 20) & (k > d), 1.0,
#                    np.where((k > 80) & (k < d), 0.0, np.nan))
#     return pd.Series(sig).ffill().iloc[-1]
#
# def signal_rsi(df_tail):
#     delta = df_tail['price'].diff()
#     gain = delta.clip(lower=0)
#     loss = (-delta).clip(lower=0)
#     avg_gain = gain.rolling(RSI_LB).mean()
#     avg_loss = loss.rolling(RSI_LB).mean()
#     rs = avg_gain / avg_loss.replace(0, np.nan)
#     rsi = 100 - (100 / (1 + rs))
#     sig = np.where(rsi < 30, 1.0,
#                    np.where(rsi > 70, 0.0, np.nan))
#     return pd.Series(sig).ffill().iloc[-1]
#
# def signal_rsi_momentum(df_tail):
#     delta = df_tail['price'].diff()
#     gain = delta.clip(lower=0)
#     loss = (-delta).clip(lower=0)
#     avg_gain = gain.rolling(RSI_MOM_LB).mean()
#     avg_loss = loss.rolling(RSI_MOM_LB).mean()
#     rs = avg_gain / avg_loss.replace(0, np.nan)
#     rsi = 100 - (100 / (1 + rs))
#     sig = np.where(rsi > 55, 1.0,
#                    np.where(rsi < 45, 0.0, np.nan))
#     return pd.Series(sig).ffill().iloc[-1]
#
# def signal_atr(df_tail):
#     prev_close = df_tail['price'].shift(1)
#     tr = pd.concat([
#         df_tail['high'] - df_tail['low'],
#         (df_tail['high'] - prev_close).abs(),
#         (df_tail['low'] - prev_close).abs()
#     ], axis=1).max(axis=1)
#     atr = tr.rolling(ATR_LB).mean()
#     upper = prev_close + ATR_SD * atr
#     lower = prev_close - ATR_SD * atr
#     sig = np.where(df_tail['price'] > upper, 1.0,
#                    np.where(df_tail['price'] < lower, 0.0, np.nan))
#     return pd.Series(sig).ffill().iloc[-1]
#
# # ─── MAIN PROCESS ──────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     # Load price data
#     df = pd.read_csv(CSV_FILE, parse_dates=['time'])
#     df = df.sort_values('time').reset_index(drop=True)
#
#     records = []
#     for i in range(ROLL_WINDOW, len(df) - 1):
#         tail = df.iloc[i-ROLL_WINDOW:i+1]
#         price_now = df.loc[i, 'price']
#         price_next = df.loc[i+1, 'price']
#
#         # Compute signals
#         rec = {
#             'time': df.loc[i, 'time'],
#             'TSMOM': signal_tsmom(tail),
#             'Stochastic': signal_stochastic(tail),
#             'RSI': signal_rsi(tail),
#             'RSI_Momentum': signal_rsi_momentum(tail),
#             'ATR': signal_atr(tail),
#             'actual': float(price_next > price_now)
#         }
#         records.append(rec)
#
#     # Build DataFrame, clean and cast
#     agg_df = pd.DataFrame(records)
#     agg_df = agg_df.dropna(subset=given_signals + ['actual'])
#     agg_df[given_signals + ['actual']] = agg_df[given_signals + ['actual']].astype(int)
#
#     # Split chronologically
#     train_df, test_df = train_test_split(agg_df, test_size=TEST_SIZE, shuffle=False)
#
#     # Save outputs
#     train_df.to_csv(TRAIN_OUT, index=False)
#     test_df.to_csv(TEST_OUT, index=False)
#     print(f"Train and test CSVs saved to '{TRAIN_OUT}' and '{TEST_OUT}'.")

#!/usr/bin/env python3
"""
Batch signal aggregation from CSV data, splitting into train/test (80/20).
Dynamically imports SignalGenerator and only processes selected signals from CSV.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from signal_class import SignalGenerator

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
CSV_FILE = 'data/BTCUSDT_1s_resample.csv'
PARAM_FILE = 'result/best_param_train_test_summary.csv'
SELECTED_FILE = 'result/selected_signal_after_wrc.csv'
TRAIN_OUT = 'result/signal_agg_train_rf.csv'
TEST_OUT = 'result/signal_agg_test_rf.csv'
TEST_SIZE = 0.2

# ─── LOAD SELECTED SIGNAL LIST ─────────────────────────────────────────────────
selected_df = pd.read_csv(SELECTED_FILE)
# assume column 'signal' contains names matching strategies in PARAM_FILE
selected_signals = selected_df['signal'].tolist()

# ─── LOAD BEST PARAMS ──────────────────────────────────────────────────────────
span_df = pd.read_csv(PARAM_FILE).set_index('strategy')

# ─── DETERMINE ROLLING WINDOW ──────────────────────────────────────────────────
def compute_rolling_window(signals):
    windows = []
    for sig in signals:
        if sig not in span_df.index:
            raise KeyError(f"Signal '{sig}' not found in parameter file")
        row = span_df.loc[sig]
        vals = []
        for col in ['short_window', 'long_window', 'lookback']:
            v = row.get(col, np.nan)
            if pd.notna(v):
                vals.append(int(v))
        if not vals:
            raise ValueError(f"No window parameter for '{sig}'")
        windows.append(max(vals))
    return max(windows) + 1

ROLL_WINDOW = compute_rolling_window(selected_signals)

# ─── HELPER ─────────────────────────────────────────────────────────────────────
def get_last(sig_series):
    arr = np.asarray(sig_series)
    return int(arr[-1])

# ─── MAIN PROCESS ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # load price data
    df = pd.read_csv(CSV_FILE, parse_dates=['time'])
    df = df.sort_values('time').reset_index(drop=True)

    records = []
    for i in range(ROLL_WINDOW, len(df) - 1):
        tail = df.iloc[i - ROLL_WINDOW:i + 1]
        price_now = df.loc[i, 'price']
        price_next = df.loc[i + 1, 'price']

        rec = {'time': df.loc[i, 'time']}
        # compute each selected signal
        window = tail.rename(columns={'price': 'last_trade_price'})
        for sig in selected_signals:
            row = span_df.loc[sig]
            series = SignalGenerator.compute_signal(
                window,
                strategy=sig,
                short_window=int(row['short_window']) if pd.notna(row.get('short_window')) else None,
                long_window=int(row['long_window']) if pd.notna(row.get('long_window')) else None,
                lookback=int(row['lookback']) if pd.notna(row.get('lookback')) else None,
                std_dev=float(row['std_dev']) if pd.notna(row.get('std_dev')) else None
            )
            rec[sig] = get_last(series)
        # actual movement
        rec['actual'] = int(price_next > price_now)
        records.append(rec)

    # build DataFrame and clean
    agg_df = pd.DataFrame(records)
    agg_df = agg_df.dropna(subset=selected_signals + ['actual'])
    agg_df[selected_signals + ['actual']] = agg_df[selected_signals + ['actual']].astype(int)

    # chronological split
    train_df, test_df = train_test_split(agg_df, test_size=TEST_SIZE, shuffle=False)

    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)
    print(f"Saved train ({len(train_df)}) and test ({len(test_df)}) to '{TRAIN_OUT}' and '{TEST_OUT}'")
