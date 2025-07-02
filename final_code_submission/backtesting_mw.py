import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

DEFAULT_TRANSACTION_COST = 0.1/10000

def generate_param_range(start, end):
    step = 5 if (end - start) < 20 else 10
    return range(start, end, step)

def append_dataset_label(metrics, label):
    for m in metrics:
        m['dataset'] = label
    return metrics

def generate_signal(df, strategy, sw=None, lw=None, lb=None, sd=None, extra=None, transaction_cost=DEFAULT_TRANSACTION_COST):
    df = df.copy()

    if strategy == 'SMA':
        df['Signal'] = np.where(
            df['last_trade_price'].rolling(window=sw).mean() >
            df['last_trade_price'].rolling(window=lw).mean(), 1, 0
        )

    elif strategy == 'EWMA':
        df['Signal'] = np.where(
            df['last_trade_price'].ewm(span=sw, adjust=False).mean() >
            df['last_trade_price'].ewm(span=lw, adjust=False).mean(), 1, 0
        )

    elif strategy == 'TSMOM':
        df['Signal'] = np.where(df['last_trade_price'] > df['last_trade_price'].shift(lb), 1, 0)

    elif strategy == 'RSI':
        delta = df['last_trade_price'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=lb).mean()
        avg_loss = loss.rolling(window=lb).mean()
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        df['Signal'] = np.where(rsi < 30, 1, np.where(rsi > 70, 0, np.nan))
        df['Signal'] = df['Signal'].ffill().fillna(0)

    elif strategy == 'RSI_Momentum':
        delta = df['last_trade_price'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=lb).mean()
        avg_loss = loss.rolling(window=lb).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df['Signal'] = np.where(rsi > 55, 1, np.where(rsi < 45, 0, np.nan))
        df['Signal'].ffill(inplace=True)

    elif strategy == 'BB':
        ma = df['last_trade_price'].rolling(window=lb).mean()
        std = df['last_trade_price'].rolling(window=lb).std()
        upper = ma + sd * std
        lower = ma - sd * std
        df['Signal'] = np.where(df['last_trade_price'] < lower, 1,
                                np.where(df['last_trade_price'] > upper, 0, np.nan))
        df['Signal'].ffill(inplace=True)

    elif strategy == 'OBV':
        df['OBV'] = (np.sign(df['last_trade_price'].diff()) * df['qty']).cumsum()
        obv_ma = df['OBV'].rolling(window=lb).mean()
        df['Signal'] = np.where(df['OBV'] > obv_ma, 1, 0)

    elif strategy == 'MACD':
        ema_fast = df['last_trade_price'].ewm(span=sw, adjust=False).mean()
        ema_slow = df['last_trade_price'].ewm(span=lw, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=lb, adjust=False).mean()
        df['Signal'] = np.where(macd_line > signal_line, 1, 0)

    elif strategy == 'Stochastic':
        low_min = df['low'].rolling(window=sw).min()
        high_max = df['high'].rolling(window=sw).max()
        df['%K'] = 100 * (df['last_trade_price'] - low_min) / (high_max - low_min)
        df['%D'] = df['%K'].rolling(window=lw).mean()
        df['Signal'] = np.where(
            (df['%K'] < 20) & (df['%K'] > df['%D']), 1,
            np.where((df['%K'] > 80) & (df['%K'] < df['%D']), 0, np.nan)
        )
        df['Signal'].ffill(inplace=True)

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

        atr = tr.rolling(window=lb).mean()

        df['Upper_Break'] = prev_close + sd * atr
        df['Lower_Break'] = prev_close - sd * atr

        df['Signal'] = np.where(close > df['Upper_Break'], 1,
                                np.where(close < df['Lower_Break'], 0, np.nan))
        df['Signal'].ffill(inplace=True)

    elif strategy == 'Donchian':
        df['Upper'] = df['high'].rolling(window=lb).max()
        df['Lower'] = df['low'].rolling(window=lb).min()
        df['Signal'] = np.where(
            df['last_trade_price'] > df['Upper'].shift(1), 1,
            np.where(df['last_trade_price'] < df['Lower'].shift(1), 0, np.nan)
        )
        df['Signal'].ffill(inplace=True)

    elif strategy == 'VWAP':
        rolling_pv = (df['last_trade_price'] * df['qty']).rolling(window=lb).sum()
        rolling_vol = df['qty'].rolling(window=lb).sum()
        df['vwap'] = rolling_pv / rolling_vol
        df['Signal'] = np.where(df['last_trade_price'] > df['vwap'], 1, 0)

    elif strategy == 'ZScore':
        rolling_mean = df['last_trade_price'].rolling(window=lb).mean()
        rolling_std = df['last_trade_price'].rolling(window=lb).std()
        df['zscore'] = (df['last_trade_price'] - rolling_mean) / rolling_std
        df['Signal'] = np.where(df['zscore'] < -sd, 1,  # Buy signal (mean reversion long)
                                np.where(df['zscore'] > sd, 0, np.nan))  # Sell signal
        df['Signal'].ffill(inplace=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    df['Position'] = df['Signal'].shift(1).diff().fillna(0)
    df['Transaction_Cost'] = df['Position'].abs() * transaction_cost

    return df



# 1) SMA Crossover
def compute_sma_metrics(data, short_range, long_range, transaction_cost=0.0005):
    metrics = []
    for sw in short_range:
        for lw in long_range:
            if sw >= lw:
                continue
            df = generate_signal(data, strategy='SMA', sw=sw, lw=lw)
            df['Next_Price'] = df['last_trade_price'].shift(-1)
            df_valid = df.dropna(subset=['Next_Price'])
            df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])

            pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
            total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

            trades = df_valid[df_valid['Position'] != 0]
            # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
            correct = 0
            total = 0
            for _, row in df_valid.iterrows():
                price_now = row['last_trade_price']
                price_next = row['Next_Price']
                if price_next != price_now:
                    true_dir = 1 if price_next > price_now else -1
                    total += 1
                    correct += (row['Position'] == true_dir)
            accuracy = correct / total if total > 0 else np.nan

            metrics.append({
                'strategy': 'SMA',
                'short_window': sw,
                'long_window': lw,
                'lookback': np.nan,
                'std_dev': np.nan,
                'PnL': total_pnl,
                'accuracy': accuracy
            })
    return metrics

# 2) EWMA Crossover
def compute_ewma_metrics(data, short_range, long_range):
    metrics = []
    for short_window in short_range:
        for long_window in long_range:
            if short_window >= long_window:
                continue

            # Generate signals
            df = generate_signal(data, strategy='EWMA', sw=short_window, lw=long_window)

            # Evaluate performance
            df['Next_Price'] = df['last_trade_price'].shift(-1)
            df_valid = df.dropna(subset=['Next_Price'])
            df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
            pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
            total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

            trades = df_valid[df_valid['Position'] != 0]
            # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
            correct = 0
            total = 0
            for _, row in df_valid.iterrows():
                price_now = row['last_trade_price']
                price_next = row['Next_Price']
                if price_next != price_now:
                    true_dir = 1 if price_next > price_now else -1
                    total += 1
                    correct += (row['Position'] == true_dir)
            accuracy = correct / total if total > 0 else np.nan

            # Store metrics
            metrics.append({
                'strategy': 'EWMA',
                'short_window': short_window,
                'long_window': long_window,
                'lookback': np.nan,
                'std_dev': np.nan,
                'PnL': total_pnl,
                'accuracy': accuracy
            })
    return metrics

# 3) TSMOM
def compute_tsmom_metrics(data, lookback_range):
    metrics = []
    for lookback in lookback_range:
        # Generate signal using the unified function
        df = generate_signal(data, strategy='TSMOM', lb=lookback)

        # Compute performance
        df['Next_Price'] = df['last_trade_price'].shift(-1)
        df_valid = df.dropna(subset=['Next_Price'])
        df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
        pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
        total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

        trades = df_valid[df_valid['Position'] != 0]
        # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
        correct = 0
        total = 0
        for _, row in df_valid.iterrows():
            price_now = row['last_trade_price']
            price_next = row['Next_Price']
            if price_next != price_now:
                true_dir = 1 if price_next > price_now else -1
                total += 1
                correct += (row['Position'] == true_dir)
        accuracy = correct / total if total > 0 else np.nan

        # Append metrics
        metrics.append({
            'strategy': 'TSMOM',
            'short_window': np.nan,
            'long_window': np.nan,
            'lookback': lookback,
            'std_dev': np.nan,
            'PnL': total_pnl,
            'accuracy': accuracy
        })
    return metrics

# 4) RSI Mean Reversion (buy when <30 (oversold)  and sell when > 70 (overbought)
def compute_rsi_metrics(data, window_range, lower=30, upper=70):
    metrics = []
    for window in window_range:
        # Generate signal via unified function
        df = generate_signal(data, strategy='RSI', lb=window)

        # Compute metrics
        df['Next_Price'] = df['last_trade_price'].shift(-1)
        df_valid = df.dropna(subset=['Next_Price'])
        df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
        pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
        total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

        trades = df_valid[df_valid['Position'] != 0]
        # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan

        correct = 0
        total = 0
        for _, row in df_valid.iterrows():
            price_now = row['last_trade_price']
            price_next = row['Next_Price']
            if price_next != price_now:
                true_dir = 1 if price_next > price_now else -1
                total += 1
                correct += (row['Position'] == true_dir)
        accuracy = correct / total if total > 0 else np.nan
        metrics.append({
            'strategy': 'RSI',
            'short_window': np.nan,
            'long_window': np.nan,
            'lookback': window,
            'std_dev': np.nan,
            'PnL': total_pnl,
            'accuracy': accuracy
        })
    return metrics


# 4.2 RSI Momentum (RSI >55 LONG AND RSI < 45 SHORT)
def compute_rsi_momentum_metrics(data, window_range):
    metrics = []
    for window in window_range:
        # Generate signal via unified function
        df = generate_signal(data, strategy='RSI_Momentum', lb=window)

        # Compute metrics
        df['Next_Price'] = df['last_trade_price'].shift(-1)
        df_valid = df.dropna(subset=['Next_Price'])
        df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
        pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
        total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

        trades = df_valid[df_valid['Position'] != 0]
        # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
        correct = 0
        total = 0
        for _, row in df_valid.iterrows():
            price_now = row['last_trade_price']
            price_next = row['Next_Price']
            if price_next != price_now:
                true_dir = 1 if price_next > price_now else -1
                total += 1
                correct += (row['Position'] == true_dir)
        accuracy = correct / total if total > 0 else np.nan
        metrics.append({
            'strategy': 'RSI_Momentum',
            'short_window': np.nan,
            'long_window': np.nan,
            'lookback': window,
            'std_dev': np.nan,
            'PnL': total_pnl,
            'accuracy': accuracy
        })
    return metrics

# 5) BB
def compute_bb_metrics(data, window_range, std_dev_range):
    metrics = []
    for window in window_range:
        for std_dev in std_dev_range:
            # Generate signal
            df = generate_signal(data, strategy='BB', lb=window, sd=std_dev)

            # Compute performance metrics
            df['Next_Price'] = df['last_trade_price'].shift(-1)
            df_valid = df.dropna(subset=['Next_Price'])
            df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
            pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
            total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

            trades = df_valid[df_valid['Position'] != 0]
            # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
            correct = 0
            total = 0
            for _, row in df_valid.iterrows():
                price_now = row['last_trade_price']
                price_next = row['Next_Price']
                if price_next != price_now:
                    true_dir = 1 if price_next > price_now else -1
                    total += 1
                    correct += (row['Position'] == true_dir)
            accuracy = correct / total if total > 0 else np.nan
            metrics.append({
                'strategy': 'BB',
                'short_window': np.nan,
                'long_window': np.nan,
                'lookback': window,
                'std_dev': std_dev,
                'PnL': total_pnl,
                'accuracy': accuracy
            })
    return metrics

# 6) OBV
def compute_obv_metrics(data, obv_ma_range):
    metrics = []
    for window in obv_ma_range:
        # Generate OBV signal using the centralized generator
        df = generate_signal(data, strategy='OBV', lb=window)

        # Compute metrics
        df['Next_Price'] = df['last_trade_price'].shift(-1)
        df_valid = df.dropna(subset=['Next_Price'])
        df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
        pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
        total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

        trades = df_valid[df_valid['Position'] != 0]
        # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
        correct = 0
        total = 0
        for _, row in df_valid.iterrows():
            price_now = row['last_trade_price']
            price_next = row['Next_Price']
            if price_next != price_now:
                true_dir = 1 if price_next > price_now else -1
                total += 1
                correct += (row['Position'] == true_dir)
        accuracy = correct / total if total > 0 else np.nan
        metrics.append({
            'strategy': 'OBV',
            'short_window': np.nan,
            'long_window': np.nan,
            'lookback': window,
            'std_dev': np.nan,
            'PnL': total_pnl,
            'accuracy': accuracy
        })
    return metrics

# 7) MACD
def compute_macd_metrics(data, fast_range, slow_range, signal_range):
    metrics = []
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            for signal in signal_range:
                # Generate signal
                df = generate_signal(data, strategy='MACD', sw=fast, lw=slow, lb=signal)

                # Compute metrics
                df['Next_Price'] = df['last_trade_price'].shift(-1)
                df_valid = df.dropna(subset=['Next_Price'])
                df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
                pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
                total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

                trades = df_valid[df_valid['Position'] != 0]
                # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
                correct = 0
                total = 0
                for _, row in df_valid.iterrows():
                    price_now = row['last_trade_price']
                    price_next = row['Next_Price']
                    if price_next != price_now:
                        true_dir = 1 if price_next > price_now else -1
                        total += 1
                        correct += (row['Position'] == true_dir)
                accuracy = correct / total if total > 0 else np.nan
                metrics.append({
                    'strategy': 'MACD',
                    'short_window': fast,
                    'long_window': slow,
                    'lookback': signal,
                    'std_dev': np.nan,
                    'PnL': total_pnl,
                    'accuracy': accuracy
                })
    return metrics

# Stochastic Oscillator
def compute_stochastic_metrics(data, k_range, d_range):
    metrics = []
    for k_period in k_range:
        for d_period in d_range:
            # Generate signal
            df = generate_signal(data, strategy='Stochastic', sw=k_period, lw=d_period)

            # Compute performance
            df['Next_Price'] = df['last_trade_price'].shift(-1)
            df_valid = df.dropna(subset=['Next_Price'])
            df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
            pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
            total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

            trades = df_valid[df_valid['Position'] != 0]
            # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
            correct = 0
            total = 0
            for _, row in df_valid.iterrows():
                price_now = row['last_trade_price']
                price_next = row['Next_Price']
                if price_next != price_now:
                    true_dir = 1 if price_next > price_now else -1
                    total += 1
                    correct += (row['Position'] == true_dir)
            accuracy = correct / total if total > 0 else np.nan
            metrics.append({
                'strategy': 'Stochastic',
                'short_window': k_period,
                'long_window': d_period,
                'lookback': np.nan,
                'std_dev': np.nan,
                'PnL': total_pnl,
                'accuracy': accuracy
            })
    return metrics

# ATR
def compute_atr_metrics(data, window_range, multiplier_range):
    metrics = []
    for window in window_range:
        for mult in multiplier_range:
            # Generate signal
            df = generate_signal(data, strategy='ATR', lb=window, sd=mult)

            # Compute metrics
            df['Next_Price'] = df['last_trade_price'].shift(-1)
            df_valid = df.dropna(subset=['Next_Price'])
            df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
            pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
            total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

            trades = df_valid[df_valid['Position'] != 0]
            # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
            correct = 0
            total = 0
            for _, row in df_valid.iterrows():
                price_now = row['last_trade_price']
                price_next = row['Next_Price']
                if price_next != price_now:
                    true_dir = 1 if price_next > price_now else -1
                    total += 1
                    correct += (row['Position'] == true_dir)
            accuracy = correct / total if total > 0 else np.nan
            metrics.append({
                'strategy': 'ATR',
                'short_window': np.nan,
                'long_window': np.nan,
                'lookback': window,
                'std_dev': mult,
                'PnL': total_pnl,
                'accuracy': accuracy
            })
    return metrics

# Donchian
def compute_donchian_metrics(data, window_range):
    metrics = []
    for window in window_range:
        # Generate signal
        df = generate_signal(data, strategy='Donchian', lb=window)

        # Compute performance metrics
        df['Next_Price'] = df['last_trade_price'].shift(-1)
        df_valid = df.dropna(subset=['Next_Price'])
        df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
        pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
        total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

        trades = df_valid[df_valid['Position'] != 0]
        # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
        correct = 0
        total = 0
        for _, row in df_valid.iterrows():
            price_now = row['last_trade_price']
            price_next = row['Next_Price']
            if price_next != price_now:
                true_dir = 1 if price_next > price_now else -1
                total += 1
                correct += (row['Position'] == true_dir)
        accuracy = correct / total if total > 0 else np.nan
        metrics.append({
            'strategy': 'Donchian',
            'short_window': np.nan,
            'long_window': np.nan,
            'lookback': window,
            'std_dev': np.nan,
            'PnL': total_pnl,
            'accuracy': accuracy
        })
    return metrics

# Rolling z-score
def compute_zscore_metrics(data, window_range, threshold=1.5):
    metrics = []
    for window in window_range:
        # Generate signal
        df = generate_signal(data, strategy='ZScore', lb=window, sd=threshold)

        # Compute performance metrics
        df['Next_Price'] = df['last_trade_price'].shift(-1)
        df_valid = df.dropna(subset=['Next_Price'])
        df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
        pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
        total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

        trades = df_valid[df_valid['Position'] != 0]
        # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
        correct = 0
        total = 0
        for _, row in df_valid.iterrows():
            price_now = row['last_trade_price']
            price_next = row['Next_Price']
            if price_next != price_now:
                true_dir = 1 if price_next > price_now else -1
                total += 1
                correct += (row['Position'] == true_dir)
        accuracy = correct / total if total > 0 else np.nan
        metrics.append({
            'strategy': 'ZScore',
            'short_window': np.nan,
            'long_window': np.nan,
            'lookback': window,
            'std_dev': threshold,
            'PnL': total_pnl,
            'accuracy': accuracy
        })
    return metrics

# VWAP
def compute_vwap_metrics(data, window_range):
    metrics = []
    for window in window_range:
        # Generate signal
        df = generate_signal(data, strategy='VWAP', lb=window)

        # Compute metrics
        df['Next_Price'] = df['last_trade_price'].shift(-1)
        df_valid = df.dropna(subset=['Next_Price'])
        df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
        pnl_series = (df_valid['Position'] * df_valid['log_return']).cumsum()
        total_pnl = pnl_series.iloc[-1] if not pnl_series.empty else np.nan

        trades = df_valid[df_valid['Position'] != 0]
        # accuracy = (np.sign(trades['log_return']) == trades['Position']).mean() if not trades.empty else np.nan
        correct = 0
        total = 0
        for _, row in df_valid.iterrows():
            price_now = row['last_trade_price']
            price_next = row['Next_Price']
            if price_next != price_now:
                true_dir = 1 if price_next > price_now else -1
                total += 1
                correct += (row['Position'] == true_dir)
        accuracy = correct / total if total > 0 else np.nan
        metrics.append({
            'strategy': 'VWAP',
            'short_window': np.nan,
            'long_window': np.nan,
            'lookback': window,
            'std_dev': np.nan,
            'PnL': total_pnl,
            'accuracy': accuracy
        })
    return metrics

def run_all_metrics(data, param_config, dataset_label):
    metrics = []
    metrics += append_dataset_label(compute_sma_metrics(data, param_config['sma_short'], param_config['sma_long']), dataset_label)
    metrics += append_dataset_label(compute_ewma_metrics(data, param_config['ewma_short'], param_config['ewma_long']), dataset_label)
    metrics += append_dataset_label(compute_tsmom_metrics(data, param_config['tsmom']), dataset_label)
    metrics += append_dataset_label(compute_rsi_metrics(data, param_config['rsi']), dataset_label)
    metrics += append_dataset_label(compute_rsi_momentum_metrics(data, param_config['rsi_mom']), dataset_label)
    metrics += append_dataset_label(compute_bb_metrics(data, param_config['bb_window'], param_config['bb_k']), dataset_label)
    metrics += append_dataset_label(compute_obv_metrics(data, param_config['obv']), dataset_label)
    metrics += append_dataset_label(compute_macd_metrics(data, param_config['macd_fast'], param_config['macd_slow'], param_config['macd_signal']), dataset_label)
    metrics += append_dataset_label(compute_stochastic_metrics(data, param_config['stoch_k'], param_config['stoch_d']), dataset_label)
    metrics += append_dataset_label(compute_atr_metrics(data, param_config['atr_window'], param_config['atr_mult']), dataset_label)
    metrics += append_dataset_label(compute_donchian_metrics(data, param_config['donchian_window']), dataset_label)
    metrics += append_dataset_label(compute_zscore_metrics(data, param_config['zscore_window']), dataset_label)
    return metrics


def extract_best_params_from_train(df_metrics):
    best_rows = []
    for strategy in df_metrics['strategy'].unique():
        # Filter for current strategy and training data
        df_strategy = df_metrics[
            (df_metrics['strategy'] == strategy) &
            (df_metrics['dataset'] == 'train')
            ].copy()

        if not df_strategy.empty:
            # Select the row with maximum PnL
            best_row = df_strategy.loc[df_strategy['PnL'].idxmax()]
            best_rows.append(best_row)

    return pd.DataFrame(best_rows)


def run_best_params_on_test(data, df_best_train):
    test_metrics = []
    for _, row in df_best_train.iterrows():
        strategy = row['strategy']
        sw = int(row['short_window']) if not pd.isna(row['short_window']) else None
        lw = int(row['long_window']) if not pd.isna(row['long_window']) else None
        lb = int(row['lookback']) if not pd.isna(row['lookback']) else None
        sd = float(row['std_dev']) if not pd.isna(row['std_dev']) else None

        if strategy == 'SMA':
            test_metrics += compute_sma_metrics(data, [sw], [lw])
        elif strategy == 'EWMA':
            test_metrics += compute_ewma_metrics(data, [sw], [lw])
        elif strategy == 'TSMOM':
            test_metrics += compute_tsmom_metrics(data, [lb])
        elif strategy == 'RSI':
            test_metrics += compute_rsi_metrics(data, [lb])
        elif strategy == 'RSI_Momentum':
            test_metrics += compute_rsi_momentum_metrics(data, [lb])
        elif strategy == 'BB':
            test_metrics += compute_bb_metrics(data, [lb], [sd])
        elif strategy == 'OBV':
            test_metrics += compute_obv_metrics(data, [lb])
        elif strategy == 'MACD':
            test_metrics += compute_macd_metrics(data, [sw], [lw], [lb])
        elif strategy == 'Stochastic':
            test_metrics += compute_stochastic_metrics(data, [sw], [lw])
        elif strategy == 'ATR':
            test_metrics += compute_atr_metrics(data, [lb], [sd])
        elif strategy == 'Donchian':
            test_metrics += compute_donchian_metrics(data, [lb])
        elif strategy == 'ZScore':
            test_metrics += compute_zscore_metrics(data, [lb])
        elif strategy == 'VWAP':
            test_metrics += compute_vwap_metrics(data, [lb])

    return append_dataset_label(test_metrics, 'test')


def test_best_params_on_test(df_best_train, df_test_metrics):
    merged = pd.merge(
        df_best_train,
        df_test_metrics,
        on=['strategy', 'short_window', 'long_window', 'lookback', 'std_dev'],
        suffixes=('_train', '_test')
    )
    result = merged[[
        'strategy',
        'short_window',
        'long_window',
        'lookback',
        'std_dev',
        'accuracy_train',
        'PnL_train',
        'accuracy_test',
        'PnL_test'
    ]].rename(columns={
        'accuracy_train': 'train_acc',
        'PnL_train': 'train_pnl',
        'accuracy_test': 'test_acc',
        'PnL_test': 'test_pnl'
    })
    return result


def generate_trade_log(data, best_params_df):
    trade_logs = []

    for _, row in best_params_df.iterrows():
        strategy = row['strategy']
        params = {
            'sw': int(row['short_window']) if not pd.isna(row['short_window']) else None,
            'lw': int(row['long_window']) if not pd.isna(row['long_window']) else None,
            'lb': int(row['lookback']) if not pd.isna(row['lookback']) else None,
            'sd': float(row['std_dev']) if not pd.isna(row['std_dev']) else None
        }

        # Generate signals with best params
        df = generate_signal(data.copy(), strategy=strategy, **params)

        # Add strategy info to each row (log every second)
        df['strategy'] = strategy
        for param in ['short_window', 'long_window', 'lookback', 'std_dev']:
            df[param] = row[param]

        trade_logs.append(df)

    return pd.concat(trade_logs).sort_index()



def plot_equity_curve(data, strategy_name, metric_row):
    try:
        df = data.copy()

        # Extract parameters safely
        params = {
            'sw': int(metric_row['short_window']) if not pd.isna(metric_row.get('short_window')) else None,
            'lw': int(metric_row['long_window']) if not pd.isna(metric_row.get('long_window')) else None,
            'lb': int(metric_row['lookback']) if not pd.isna(metric_row.get('lookback')) else None,
            'sd': float(metric_row['std_dev']) if not pd.isna(metric_row.get('std_dev')) else None
        }

        # Generate signals
        df = generate_signal(df, strategy=strategy_name, **{k: v for k, v in params.items() if v is not None})

        # Calculate strategy returns
        df['Next_Price'] = df['last_trade_price'].shift(-1)
        df_valid = df.dropna(subset=['Next_Price']).copy()
        df_valid['log_return'] = np.log(df_valid['Next_Price'] / df_valid['last_trade_price'])
        df_valid['PnL'] = df_valid['Position'] * df_valid['log_return']
        df_valid['Equity'] = df_valid['PnL'].cumsum()

        # Buy & Hold benchmark
        baseline = df.copy()
        baseline['Next_Price'] = baseline['last_trade_price'].shift(-1)
        baseline = baseline.dropna(subset=['Next_Price']).copy()
        baseline['log_return'] = np.log(baseline['Next_Price'] / baseline['last_trade_price'])
        baseline['Equity'] = baseline['log_return'].cumsum()

        # Plot
        plt.figure(figsize=(14, 6))
        plt.plot(df_valid.index, df_valid['Equity'], label=f'{strategy_name} Strategy')
        plt.plot(baseline.index, baseline['Equity'], label='Buy & Hold', linestyle='--')
        plt.title(f'Equity Curve: {strategy_name} vs Buy & Hold\nParams: {params}')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Log PnL')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error plotting {strategy_name}: {str(e)}")



def main():
    os.makedirs('../result', exist_ok=True)
    data = pd.read_csv('data/BTCUSDT_1s_resample.csv')
    if 'price' in data.columns:
        data.rename(columns={'price': 'last_trade_price'}, inplace=True)
    if 'last_trade_price' not in data.columns:
        raise ValueError('Missing price column.')
    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'], unit='ms', errors='ignore')

    data = data.sort_values('time', ascending=True)
    split_idx = int(len(data) * 0.8)
    train_data, test_data = data.iloc[:split_idx], data.iloc[split_idx:]
    train_data2 = train_data.copy()
    train_data2.rename(columns={'last_trade_price': 'price'}, inplace=True)
    train_data2.to_csv('result/train_backtesting.csv', index=False)

    test_data2 = test_data.copy()
    test_data2.rename(columns={'last_trade_price': 'price'}, inplace=True)
    test_data2.to_csv('result/test_backtesting.csv', index=False)


    baseline = test_data.copy()
    baseline['Next_Price'] = baseline['last_trade_price'].shift(-1)
    baseline.dropna(subset=['Next_Price'], inplace=True)
    baseline['log_return'] = np.log(baseline['Next_Price'] / baseline['last_trade_price'])
    market_total_pnl = baseline['log_return'].cumsum().iloc[-1]


    # Define the parameters range for backtesting
    param_config = {
        # SMA and EWMA crossovers
        'sma_short': range(3, 15, 2),
        'sma_long': range(20, 60, 5),
        'ewma_short': range(3, 15, 2),
        'ewma_long': range(20, 60, 5),

        'tsmom': range(5, 60, 5),
        'rsi': range(5, 21, 2),
        'rsi_mom': range(5, 21, 2),

        'bb_window': range(10, 30, 5),
        'bb_k': [1.5, 2.0, 2.5],

        'obv': generate_param_range(5, 40),

        'macd_fast': range(3, 9, 2), # EMA 3-7
        'macd_slow': range(9, 21, 2), #EMA 9-19
        'macd_signal': range(3, 10, 2),

        'stoch_k': range(5, 13, 2),
        'stoch_d': range(2, 5),

        'atr_window': range(10, 30, 5),
        'atr_mult': [1.0, 1.5, 2.0],

        'donchian_window': generate_param_range(5, 30),

        'zscore_window': generate_param_range(10, 30),
        'vwap': generate_param_range(3, 30)
    }


    print('Running metrics on train datasets...')
    all_metrics = run_all_metrics(train_data, param_config, 'train')
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics['market_PnL'] = market_total_pnl
    df_metrics.to_csv('result/combined_strategy_metrics_train.csv', index=False)

    df_best_train = extract_best_params_from_train(df_metrics)
    test_metrics = run_best_params_on_test(test_data, df_best_train)
    df_test_metrics = pd.DataFrame(test_metrics)

    df_metrics = pd.concat([df_metrics, df_test_metrics], ignore_index=True)
    df_metrics.to_csv('result/combined_strategy_metrics_full.csv', index=False)

    df_summary = test_best_params_on_test(df_best_train, df_test_metrics)
    df_summary.to_csv('result/best_param_train_test_summary.csv', index=False)
    print('Saved best_param_train_test_summary.csv')
    print(df_summary)

    # Best param trade log
    best_params = extract_best_params_from_train(df_metrics)
    trade_log = generate_trade_log(test_data, best_params)
    trade_log.to_csv('best_params_trade_log.csv', index=True)

    for _, row in df_best_train.iterrows():
        strategy = row['strategy']
        print(f"Plotting {strategy} on test set...")
        plot_equity_curve(test_data, strategy, row)



if __name__ == '__main__':
    main()
