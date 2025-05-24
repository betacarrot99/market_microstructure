import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

start_time = time.time()

def sma_strategy_with_tuning(data, short_range, long_range):
    best_params = {'short_window': None, 'long_window': None, 'accuracy': -np.inf}
    results = pd.DataFrame()

    for short_window in short_range:
        for long_window in long_range:
            if short_window >= long_window:
                continue

            df = data.copy()
            df['shorter_SMA'] = df['last_trade_price'].rolling(window=short_window).mean()
            df['longer_SMA'] = df['last_trade_price'].rolling(window=long_window).mean()
            df['Signal'] = np.where(df['shorter_SMA'] > df['longer_SMA'], 1, 0)

            # Actual market direction
            df['Market_Direction'] = np.where(df['last_trade_price'].shift(-1) > df['last_trade_price'], 1, 0)

            # Accuracy score
            df['Correct'] = df['Signal'] == df['Market_Direction']
            accuracy = df['Correct'].mean()

            if accuracy > best_params['accuracy']:
                best_params = {
                    'short_window': short_window,
                    'long_window': long_window,
                    'accuracy': accuracy
                }
                results = df.copy()

    return best_params, results

def log_best_params(file_path, timestamp, short_window, long_window, accuracy, processing_time, input_start_date, input_end_date, signal_method):
    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "short_window": short_window,
        "long_window": long_window,
        "accuracy": accuracy,
        "processing_time" : processing_time,
        "input_date" : input_start_date,
        "input_end_date" : input_end_date,
        "method": signal_method
    }])
    if os.path.exists(file_path):
        new_entry.to_csv(file_path, mode='a', index=False, header=False)
    else:
        new_entry.to_csv(file_path, index=False, header=True)


# Load and prepare data
data2 = pd.read_csv('data/BTCUSDT-trades-2025-05-20.csv')
# data1 = pd.read_csv('data/BTCUSDT-trades-2025-05-18.csv')
# data = pd.concat([data1, data2])
data = data2
data['time'] = pd.to_datetime(data['time'], unit='ms')
data = data.rename(columns={'price': 'last_trade_price'})

# Train-test split
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Find best params on training data
best_params, train_results = sma_strategy_with_tuning(
    train_data,
    short_range=range(5, 20),
    long_range=range(50, 200)
)


print("Best Parameters (by accuracy):", best_params)
print(f"Train signal accuracy: {best_params['accuracy']:.2%}")

# Apply best params to test data
short = best_params['short_window']
long = best_params['long_window']

test_data = test_data.copy()
test_data['shorter_SMA'] = test_data['last_trade_price'].rolling(window=short).mean()
test_data['longer_SMA'] = test_data['last_trade_price'].rolling(window=long).mean()
test_data['Signal'] = np.where(test_data['shorter_SMA'] > test_data['longer_SMA'], 1, 0)
test_data['Market_Direction'] = np.where(test_data['last_trade_price'].shift(-1) > test_data['last_trade_price'], 1, 0)
test_data['Correct'] = test_data['Signal'] == test_data['Market_Direction']

test_accuracy = test_data['Correct'].mean()
print(f"Test signal accuracy: {test_accuracy:.2%}")

# Combine results for overall accuracy
combined_results = pd.concat([train_results, test_data])
overall_accuracy = combined_results['Correct'].mean()
print(f"Overall signal accuracy: {overall_accuracy:.2%}")

end_time = time.time()
processing_time = round(end_time - start_time, 2)
# Log best training parameters only (test is for evaluation)
log_best_params(
    file_path='result/sma_param_log3.csv',
    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    short_window=short,
    long_window=long,
    accuracy=best_params['accuracy'],
    processing_time = processing_time,
    input_start_date = data['time'].min().date(),
    input_end_date = data['time'].max().date(),
    signal_method = "SMA"
)
print(f"Processing time:{processing_time} seconds")

# # Calculate rolling accuracy over time
# combined_results = combined_results.sort_values('time')
# combined_results['Rolling_Accuracy'] = combined_results['Correct'].rolling(window=1000).mean()

# # Plot rolling accuracy
# plt.figure(figsize=(12, 5))
# plt.plot(combined_results['time'], combined_results['Rolling_Accuracy'], color='blue', label='Rolling Accuracy (window=1000)')
# plt.xlabel("Time")
# plt.ylabel("Accuracy")
# plt.title("Rolling Signal Accuracy Over Time")
# plt.ylim(0, 1)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


