import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

start_time = time.time()

def tsmom_strategy_with_tuning(data, lookback_range):
    best_params = {'lookback': None, 'accuracy': -np.inf}
    results = pd.DataFrame()

    for lookback in lookback_range:
        df = data.copy()
        df['Signal'] = np.where(df['last_trade_price'] > df['last_trade_price'].shift(lookback), 1, 0)
        df['Market_Direction'] = np.where(df['last_trade_price'].shift(-1) > df['last_trade_price'], 1, 0)
        df['Correct'] = df['Signal'] == df['Market_Direction']
        accuracy = df['Correct'].mean()

        if accuracy > best_params['accuracy']:
            best_params = {'lookback': lookback, 'accuracy': accuracy}
            results = df.copy()

    return best_params, results

def log_best_params(file_path, timestamp, lookback, accuracy, processing_time, input_start_date, input_end_date, signal_method):
    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "lookback": lookback,
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

# Load and prepare data
data = pd.read_csv('data/BTCUSDT-trades-2025-05-20.csv')
data['time'] = pd.to_datetime(data['time'], unit='ms')
data = data.rename(columns={'price': 'last_trade_price'})

# Train-test split
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Find best TSMOM lookback on training data
best_params, train_results = tsmom_strategy_with_tuning(train_data, lookback_range=range(1, 1000))

print("Best TSMOM lookback period:", best_params['lookback'])
print(f"Train signal accuracy: {best_params['accuracy']:.2%}")

# Apply to test data
lookback = best_params['lookback']
test_data = test_data.copy()
test_data['Signal'] = np.where(test_data['last_trade_price'] > test_data['last_trade_price'].shift(lookback), 1, 0)
test_data['Market_Direction'] = np.where(test_data['last_trade_price'].shift(-1) > test_data['last_trade_price'], 1, 0)
test_data['Correct'] = test_data['Signal'] == test_data['Market_Direction']
test_accuracy = test_data['Correct'].mean()
print(f"Test signal accuracy: {test_accuracy:.2%}")

# Combine results
combined_results = pd.concat([train_results, test_data])
overall_accuracy = combined_results['Correct'].mean()
print(f"Overall signal accuracy: {overall_accuracy:.2%}")

# Log best training parameters
end_time = time.time()
processing_time = round(end_time - start_time, 2)
log_best_params(
    file_path='result/sma_param_log3.csv',
    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    lookback=lookback,
    accuracy=best_params['accuracy'],
    processing_time=processing_time,
    input_start_date=data['time'].min().date(),
    input_end_date=data['time'].max().date(),
    signal_method="TSMOM"
)
print(f"Processing time: {processing_time} seconds")
