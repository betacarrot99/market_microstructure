import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

start_time = time.time()

def random_benchmark_strategy(data):
    df = data.copy()

    # Generate random signals (0 or 1)
    df['Signal'] = np.random.randint(0, 2, size=len(df))

    # Actual market direction
    df['Market_Direction'] = np.where(df['last_trade_price'].shift(-1) > df['last_trade_price'], 1, 0)

    # Accuracy score
    df['Correct'] = df['Signal'] == df['Market_Direction']
    accuracy = df['Correct'].mean()

    return accuracy, df

def log_benchmark_results(file_path, timestamp, accuracy, processing_time, input_start_date, input_end_date):
    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "short_window": "random",
        "long_window": "random",
        "accuracy": accuracy,
        "processing_time": processing_time,
        "input_date": input_start_date,
        "input_end_date": input_end_date
    }])
    if os.path.exists(file_path):
        new_entry.to_csv(file_path, mode='a', index=False, header=False)
    else:
        new_entry.to_csv(file_path, index=False, header=True)

# Load and prepare data
data = pd.read_csv('data/BTCUSDT-trades-2025-05-21.csv')
data['time'] = pd.to_datetime(data['time'], unit='ms')
data = data.rename(columns={'price': 'last_trade_price'})

# Train-test split
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Train random benchmark model
train_accuracy, train_results = random_benchmark_strategy(train_data)
print(f"Train random benchmark accuracy: {train_accuracy:.2%}")

# Test random benchmark model
test_accuracy, test_results = random_benchmark_strategy(test_data)
print(f"Test random benchmark accuracy: {test_accuracy:.2%}")

# Combine results for overall accuracy
combined_results = pd.concat([train_results, test_results])
overall_accuracy = combined_results['Correct'].mean()
print(f"Overall random benchmark accuracy: {overall_accuracy:.2%}")

# Log results
end_time = time.time()
processing_time = round(end_time - start_time, 2)

log_benchmark_results(
    file_path='result/sma_param_log2.csv',
    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    accuracy=overall_accuracy,
    processing_time=processing_time,
    input_start_date=data['time'].min().date(),
    input_end_date=data['time'].max().date()
)

print(f"Processing time: {processing_time} seconds")
