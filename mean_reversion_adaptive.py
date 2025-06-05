import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

# Start timer
start_time = time.time()

def mean_reversion_strategy_with_tuning(data, std_range, mean_range):
    """
    Tune over combinations of (std_window, mean_window).
    - rolling_mean = price.rolling(window=mean_window).mean()
    - rolling_std  = price.rolling(window=std_window).std()
    - z_score      = (price - rolling_mean) / rolling_std
    Signal = 1 if z_score < 0 (price below its rolling mean), else 0.
    Market_Direction = 1 if next_bar_price > current_price, else 0.
    Returns the best (std_window, mean_window, accuracy) and the DataFrame for those params.
    """
    best_params = {'std_window': None, 'mean_window': None, 'accuracy': -np.inf}
    best_df = pd.DataFrame()

    for std_window in std_range:
        for mean_window in mean_range:
            if std_window >= mean_window:
                continue

            df = data.copy()
            price = df['last_trade_price']

            # 1) rolling mean over mean_window
            df['rolling_mean'] = price.rolling(window=mean_window).mean()

            # 2) rolling std over std_window
            df['rolling_std'] = price.rolling(window=std_window).std()

            # 3) z_score
            df['z_score'] = (price - df['rolling_mean']) / df['rolling_std']

            # drop rows until both rolling_mean & rolling_std are available
            df = df.dropna(subset=['z_score']).copy()

            # 4) mean-reversion signal
            df['Signal'] = np.where(df['z_score'] < 0, 1, 0)

            # 5) next-bar market direction
            df['Market_Direction'] = np.where(
                df['last_trade_price'].shift(-1) > df['last_trade_price'], 1, 0
            )

            # 6) accuracy
            df['Correct'] = (df['Signal'] == df['Market_Direction'])
            accuracy = df['Correct'].mean()

            if accuracy > best_params['accuracy']:
                best_params = {
                    'std_window': std_window,
                    'mean_window': mean_window,
                    'accuracy': accuracy
                }
                best_df = df.copy()

    return best_params, best_df

def log_best_params(file_path, timestamp, std_window, mean_window, accuracy, processing_time, input_start_date, input_end_date, signal_method):
    """
    Append (or create) a CSV log at file_path with columns:
    timestamp, std_window, mean_window, accuracy, processing_time, input_date, input_end_date, method
    """
    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "std_window": std_window,
        "mean_window": mean_window,
        "accuracy": accuracy,
        "processing_time": processing_time,
        "input_date": input_start_date,
        "input_end_date": input_end_date,
        "method": signal_method
    }])

    if os.path.exists(file_path):
        new_entry.to_csv(file_path, mode='a', index=False, header=False)
    else:
        new_entry.to_csv(file_path, index=False, header=True)

# ─── Load and prepare data ─────────────────────────────────────────────────────
data2 = pd.read_csv('data/BTCUSDT-trades-2025-05-20.csv')
data = data2.copy()
data['time'] = pd.to_datetime(data['time'], unit='ms')
data = data.rename(columns={'price': 'last_trade_price'})

# ─── Train-test split ──────────────────────────────────────────────────────────
train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size].copy()
test_data = data.iloc[train_size:].copy()

# ─── Find best params on training data (mean reversion tuning) ─────────────────
best_params, train_results = mean_reversion_strategy_with_tuning(
    train_data,
    std_range=range(5, 20),    # e.g. std lookbacks 5 to 19
    mean_range=range(50, 200)  # e.g. mean lookbacks 50 to 199
)

print("Best Parameters (by accuracy):", best_params)
print(f"Train signal accuracy: {best_params['accuracy']:.2%}")

# ─── Apply best params to test data ────────────────────────────────────────────
std_win  = best_params['std_window']
mean_win = best_params['mean_window']

test_df = test_data.copy()
price_test = test_df['last_trade_price']

# Compute rolling mean & std on test set
test_df['rolling_mean'] = price_test.rolling(window=mean_win).mean()
test_df['rolling_std']  = price_test.rolling(window=std_win).std()
test_df['z_score']      = (price_test - test_df['rolling_mean']) / test_df['rolling_std']

# Drop rows until z_score is available
test_df = test_df.dropna(subset=['z_score']).copy()

# Generate mean-reversion signal
test_df['Signal'] = np.where(test_df['z_score'] < 0, 1, 0)

# Next-bar market direction
test_df['Market_Direction'] = np.where(
    test_df['last_trade_price'].shift(-1) > test_df['last_trade_price'], 1, 0
)

test_df['Correct'] = (test_df['Signal'] == test_df['Market_Direction'])
test_accuracy = test_df['Correct'].mean()
print(f"Test signal accuracy: {test_accuracy:.2%}")

# ─── Overall accuracy (train + test) ───────────────────────────────────────────
combined = pd.concat([train_results, test_df])
overall_accuracy = combined['Correct'].mean()
print(f"Overall signal accuracy: {overall_accuracy:.2%}")

# ─── Log best parameters ───────────────────────────────────────────────────────
end_time = time.time()
processing_time = round(end_time - start_time, 2)

log_best_params(
    file_path='result/sma_param_log3.csv',
    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    std_window=std_win,
    mean_window=mean_win,
    accuracy=best_params['accuracy'],
    processing_time=processing_time,
    input_start_date=data['time'].min().date(),
    input_end_date=data['time'].max().date(),
    signal_method="MeanReversion"
)

print(f"Processing time: {processing_time} seconds")

# ─── (Optional) Plot rolling accuracy over time ─────────────────────────────────
# Uncomment below if you want to visualize rolling accuracy with a 1000-bar window.
#
combined_sorted = combined.sort_values('time')
combined_sorted['Rolling_Accuracy'] = combined_sorted['Correct'].rolling(window=1000).mean()

plt.figure(figsize=(12, 5))
plt.plot(combined_sorted['time'], combined_sorted['Rolling_Accuracy'], label='Rolling Accuracy (window=1000)')
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.title("Rolling Signal Accuracy Over Time")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
