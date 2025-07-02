import pandas as pd
import numpy as np

# from Archived.wrc_original import TICK_DATA_CSV

#--- Configuration ---
TICK_DATA_CSV = 'data/BTCUSDT_1min_resample.csv'
# TICK_DATA_CSV = 'result/train_backtesting.csv'
PARAMS_CSV    = 'result/combined_strategy_metrics_train.csv'
OUTPUT_CSV    = 'result/strategy_log_return.csv'


def compute_pnl_series(data, strategy, short_window=None, long_window=None, lookback=None, std_dev=None):
    df = data.copy()
    # generate signals
    if strategy == 'SMA':
        df['ind_short'] = df['last_trade_price'].rolling(window=short_window).mean()
        df['ind_long']  = df['last_trade_price'].rolling(window=long_window).mean()
        df['Signal']    = np.where(df['ind_short'] > df['ind_long'], 1, 0)

    elif strategy == 'EWMA':
        df['ind_short'] = df['last_trade_price'].ewm(span=short_window, adjust=False).mean()
        df['ind_long']  = df['last_trade_price'].ewm(span=long_window, adjust=False).mean()
        df['Signal']    = np.where(df['ind_short'] > df['ind_long'], 1, 0)

    elif strategy == 'TSMOM':
        df['Signal']    = np.where(
            df['last_trade_price'] > df['last_trade_price'].shift(lookback), 1, 0
        )

    elif strategy == 'RSI':
        delta = df['last_trade_price'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=lookback).mean()
        avg_loss = loss.rolling(window=lookback).mean()
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        df['Signal'] = np.where(rsi < 30, 1, np.where(rsi > 70, 0, np.nan))
        df['Signal'] = df['Signal'].ffill().fillna(0)

    elif strategy == 'RSI_Momentum':
        delta = df['last_trade_price'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=lookback).mean()
        avg_loss = loss.rolling(window=lookback).mean()
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        df['Signal'] = np.where(rsi > 55, 1, np.where(rsi < 45, 0, np.nan))
        df['Signal'] = df['Signal'].ffill().fillna(0)

    elif strategy == 'BB':
        ma = df['last_trade_price'].rolling(window=lookback).mean()
        std = df['last_trade_price'].rolling(window=lookback).std()
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        df['Signal'] = np.where(df['last_trade_price'] < lower, 1,
                                np.where(df['last_trade_price'] > upper, 0, np.nan))
        df['Signal'] = df['Signal'].ffill().fillna(0)

    elif strategy == 'OBV':
        # if volume column is named 'qty', otherwise assume constant 1
        vol = df['qty'] if 'qty' in df.columns else 1
        df['OBV'] = (np.sign(df['last_trade_price'].diff()) * vol).cumsum()
        obv_ma = df['OBV'].rolling(window=lookback).mean()
        df['Signal'] = np.where(df['OBV'] > obv_ma, 1, 0)

    elif strategy == 'MACD':
        ema_fast = df['last_trade_price'].ewm(span=short_window, adjust=False).mean()
        ema_slow = df['last_trade_price'].ewm(span=long_window, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=lookback, adjust=False).mean()
        df['Signal'] = np.where(macd_line > signal_line, 1, 0)

    elif strategy == 'Stochastic':
        low_min = df['low'].rolling(window=short_window).min()
        high_max = df['high'].rolling(window=short_window).max()
        k = 100 * (df['last_trade_price'] - low_min) / (high_max - low_min)
        d = k.rolling(window=long_window).mean()
        df['Signal'] = np.where((k < 20) & (k > d), 1,
                                np.where((k > 80) & (k < d), 0, np.nan))
        df['Signal'] = df['Signal'].ffill().fillna(0)

    elif strategy == 'ATR':
        high = df['high']
        low = df['low']
        close = df['last_trade_price']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=lookback).mean()
        upper_break = prev_close + std_dev * atr
        lower_break = prev_close - std_dev * atr
        df['Signal'] = np.where(close > upper_break, 1,
                                np.where(close < lower_break, 0, np.nan))
        df['Signal'] = df['Signal'].ffill().fillna(0)

    elif strategy == 'Donchian':
        upper = df['high'].rolling(window=lookback).max()
        lower = df['low'].rolling(window=lookback).min()
        df['Signal'] = np.where(df['last_trade_price'] > upper.shift(1), 1,
                                np.where(df['last_trade_price'] < lower.shift(1), 0, np.nan))
        df['Signal'] = df['Signal'].ffill().fillna(0)

    elif strategy == 'ZScore':
        rolling_mean = df['last_trade_price'].rolling(window=lookback).mean()
        rolling_std = df['last_trade_price'].rolling(window=lookback).std()
        zscore = (df['last_trade_price'] - rolling_mean) / rolling_std
        df['Signal'] = np.where(zscore < -std_dev, 1,
                                np.where(zscore > std_dev, 0, np.nan))
        df['Signal'] = df['Signal'].ffill().fillna(0)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # make positions 1-step look-forward, compute log-returns
    df['Position']   = df['Signal'].shift(1).diff().fillna(0)
    df['Next_Price'] = df['last_trade_price'].shift(-1)
    df['log_return'] = np.log(df['Next_Price'] / df['last_trade_price'])
    return df['Position'] * df['log_return']
# ensure no missing grips in the signal
#     df['Signal'] = df['Signal'].ffill().fillna(0)
#
#     # --- 2) shift the signal forward one bar to get your position (hold) ---
#     df['Position'] = df['Signal'].shift(1).fillna(0)
#
#     # --- 3) compute next‐bar log‐returns and drop the last NaN row ---
#     df['Next_Price'] = df['last_trade_price'].shift(-1)
#     df = df.dropna(subset=['Next_Price'])
#     df['log_return'] = np.log(df['Next_Price'] / df['last_trade_price'])
#
#     # --- 4) raw PnL per bar under a buy-and-hold‐in-signal policy ---
#     pnl = df['Position'] * df['log_return']
#
#     # --- 5) reindex back to the original full length (fill non‐trade bars with 0) ---
#     pnl_full = pnl.reindex(data.index).fillna(0)
#
#     return pnl_full

if __name__ == '__main__':
    # 1) load tick data
    tick = pd.read_csv(TICK_DATA_CSV)
    if 'price' in tick.columns:
        tick.rename(columns={'price': 'last_trade_price'}, inplace=True)
    elif 'last_trade_price' not in tick.columns:
        raise ValueError('CSV must contain price or last_trade_price column')
    tick['timestamp'] = pd.to_datetime(tick['time'])

    # 2) load parameters
    params = pd.read_csv(PARAMS_CSV)

    # 3) prepare output DataFrame
    out = pd.DataFrame({
        'timestamp': tick['timestamp'],
        'last_trade_price': tick['last_trade_price']
    })

    # 4) compute each strategy's PnL series
    for _, row in params.iterrows():
        strat = row['strategy']
        sw = int(row['short_window']) if not pd.isna(row['short_window']) else None
        lw = int(row['long_window']) if not pd.isna(row['long_window']) else None
        lb = int(row['lookback']) if not pd.isna(row['lookback']) else None
        sd = float(row['std_dev']) if 'std_dev' in row and not pd.isna(row['std_dev']) else None

        # column name
        if strat in ['SMA', 'EWMA']:
            col = f"{strat}_sw{sw}_lw{lw}"
        elif strat == 'MACD':
            col = f"{strat}_sw{sw}_lw{lw}_lb{lb}"
        elif strat in ['BB', 'ATR', 'ZScore']:
            col = f"{strat}_lb{lb}_sd{sd}"
        elif strat == 'Stochastic':
            col = f"{strat}_sw{sw}_lw{lw}"
        else:
            # strategies with only lookback
            col = f"{strat}_lb{lb}"

        out[col] = compute_pnl_series(tick, strat, sw, lw, lb, sd)

    # 5) save output
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved full-series PnL to {OUTPUT_CSV}, shape = {out.shape}")
    print(out)
