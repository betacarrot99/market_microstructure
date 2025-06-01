import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

start_time = time.time()


def logistic_regression_with_lag_tuning(data, lag_range):
    best_params = {'lag': None, 'accuracy': -np.inf}
    results = pd.DataFrame()

    for lag in lag_range:
        df = data.copy()
        for i in range(1, lag + 1):
            df[f'lag_{i}'] = df['last_trade_price'].shift(i)

        df['Target'] = np.where(df['last_trade_price'].shift(-1) > df['last_trade_price'], 1, 0)
        df = df.dropna()

        X = df[[f'lag_{i}' for i in range(1, lag + 1)]]
        y = df['Target']

        # Train/test split inside training set
        split_idx = int(0.8 * len(df))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)

        if accuracy > best_params['accuracy']:
            best_params = {'lag': lag, 'accuracy': accuracy, 'model': model}
            results = df.copy()

    return best_params, results


def log_best_params(file_path, timestamp, lag, accuracy, processing_time, input_start_date, input_end_date,
                    signal_method):
    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "short_window": lag,
        "long_window": "",
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

# Train model to find best lag
best_params, train_results = logistic_regression_with_lag_tuning(train_data, lag_range=range(1, 50))
print("Best logistic lag:", best_params['lag'])
print(f"Train signal accuracy: {best_params['accuracy']:.2%}")

# Apply to test data
model = best_params['model']
lag = best_params['lag']

test_data = test_data.copy()
for i in range(1, lag + 1):
    test_data[f'lag_{i}'] = test_data['last_trade_price'].shift(i)
test_data['Target'] = np.where(test_data['last_trade_price'].shift(-1) > test_data['last_trade_price'], 1, 0)
test_data = test_data.dropna()

X_test = test_data[[f'lag_{i}' for i in range(1, lag + 1)]]
y_test = test_data['Target']
test_preds = model.predict(X_test)

test_data['Signal'] = test_preds
test_data['Correct'] = test_data['Signal'] == test_data['Target']
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
    lag=lag,
    accuracy=best_params['accuracy'],
    processing_time=processing_time,
    input_start_date=data['time'].min().date(),
    input_end_date=data['time'].max().date(),
    signal_method="LogReg"
)
# Save the best logistic regression model to file
import joblib
joblib.dump(model, 'result/logreg_model.pkl')

print(f"Processing time: {processing_time} seconds")
