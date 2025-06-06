import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Point to the actual file location
CSV_FILE = os.path.expanduser('/market_microstructure/live_data.csv')
ROLLING_WINDOW = 50  # Must match max indicator period used
CHECK_INTERVAL = 1  # Check every 2 seconds

accuracy_tracker = {
    'ATR_signal': {'correct': 0, 'total': 0},
    'Donchian_signal': {'correct': 0, 'total': 0},
    'Chaikin_signal': {'correct': 0, 'total': 0}
}

prev_row = None

# === Volatility Indicator Calculations ===

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_donchian(df, period=20):
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    return upper, lower

def calculate_chaikin_volatility(df, period=10):
    hl_range = df['high'] - df['low']
    ema = hl_range.ewm(span=period, adjust=False).mean()
    return (ema - ema.shift(period)) / ema.shift(period) * 100

# === Signal Logic ===

def generate_volatility_signals(df):
    df = df.copy()
    df['ATR'] = calculate_atr(df)
    df['Donchian_upper'], df['Donchian_lower'] = calculate_donchian(df)
    df['Chaikin_vol'] = calculate_chaikin_volatility(df)
    df['price'] = df['close']

    df['ATR_signal'] = 'neutral'
    df['Donchian_signal'] = 'neutral'
    df['Chaikin_signal'] = 'neutral'

    i = -1

    if df['ATR'].iloc[i] > df['ATR'].iloc[i - 1]:
        if df['price'].iloc[i] > df['price'].iloc[i - 1]:
            df.loc[df.index[i], 'ATR_signal'] = 'up'
        elif df['price'].iloc[i] < df['price'].iloc[i - 1]:
            df.loc[df.index[i], 'ATR_signal'] = 'down'

    if df['price'].iloc[i] > df['Donchian_upper'].iloc[i]:
        df.loc[df.index[i], 'Donchian_signal'] = 'up'
    elif df['price'].iloc[i] < df['Donchian_lower'].iloc[i]:
        df.loc[df.index[i], 'Donchian_signal'] = 'down'

    if df['Chaikin_vol'].iloc[i] > df['Chaikin_vol'].iloc[i - 1]:
        if df['price'].iloc[i] > df['price'].iloc[i - 1]:
            df.loc[df.index[i], 'Chaikin_signal'] = 'up'
        elif df['price'].iloc[i] < df['price'].iloc[i - 1]:
            df.loc[df.index[i], 'Chaikin_signal'] = 'down'

    return df.iloc[-1]

# === Main Loop ===

while True:
    try:
        if not os.path.exists(CSV_FILE):
            logging.warning(f"{CSV_FILE} does not exist yet.")
            time.sleep(CHECK_INTERVAL)
            continue

        df = pd.read_csv(CSV_FILE)
        if len(df) < ROLLING_WINDOW + 1:
            logging.info(f"Waiting for more data... ({len(df)} rows so far)")
            time.sleep(CHECK_INTERVAL)
            continue

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df[['price', 'qty']] = df[['price', 'qty']].astype(float)

        # Convert trade rows into OHLC structure (you can improve this later)
        df['open'] = df['high'] = df['low'] = df['close'] = df['price']

        df = df.tail(ROLLING_WINDOW + 1)
        latest_row = generate_volatility_signals(df)
        current_price = latest_row['price']

        if prev_row is not None:
            for ind in ['ATR_signal', 'Donchian_signal', 'Chaikin_signal']:
                signal = prev_row[ind]
                prev_price = prev_row['price']
                accurate = False

                if signal == 'up':
                    accurate = current_price > prev_price
                elif signal == 'down':
                    accurate = current_price < prev_price
                else:
                    change = abs(current_price - prev_price) / prev_price
                    accurate = change < 0.000005  # 0.0005%

                accuracy_tracker[ind]['total'] += 1
                if accurate:
                    accuracy_tracker[ind]['correct'] += 1

        def acc_str(ind):
            correct = accuracy_tracker[ind]['correct']
            total = accuracy_tracker[ind]['total']
            pct = (correct / total * 100) if total > 0 else 0
            return f"{latest_row[ind]:<8} > accuracy: {pct:.2f}%"

        logging.info(
            f"Price: {current_price:.2f} | "
            f"ATR: {acc_str('ATR_signal')}, "
            f"Donchian: {acc_str('Donchian_signal')}, "
            f"Chaikin: {acc_str('Chaikin_signal')}"
        )

        prev_row = latest_row

    except Exception as e:
        logging.warning(f"Error in processing: {e}")

    time.sleep(CHECK_INTERVAL)