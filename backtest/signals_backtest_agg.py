# signal_backtest_agg.py
import pandas as pd
import time
import json

from predictor.signal_manager import SignalManager
from backtest.backtester import SignalBacktester
from predictor.aggregator.aggregator_manager import AggregatorManager
# from predictor.aggregator.xgboost_model import XGBoostAggregator

# Load historical trade data (resampled)
data_file = "../data/btcusdt_400ms_resample.csv"
df = pd.read_csv(data_file, parse_dates=["timestamp"])

threshold = 0.00001  # 0.001%

# Split into 80% train / 20% test
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

start_time = time.time()

# Generate individual signals
manager = SignalManager()
opt_params = manager.optimize_all(train_df)
train_signal_df = manager.generate_all_signals(train_df, opt_params)
test_signal_df = manager.generate_all_signals(test_df, opt_params)


# Generate aggregated signals
agg_methods = ["equal_weight", "accuracy_weight", "majority_vote", "xgboost_model"]
agg_manager = AggregatorManager(agg_methods)

train_agg_signals = [
    agg_manager.aggregate(row.to_dict()) for _, row in train_signal_df.iterrows()
]
test_agg_signals = [
    agg_manager.aggregate(row.to_dict()) for _, row in test_signal_df.iterrows()
]

train_signal_df = pd.concat([train_signal_df.reset_index(drop=True), pd.DataFrame(train_agg_signals)], axis=1)
test_signal_df = pd.concat([test_signal_df.reset_index(drop=True), pd.DataFrame(test_agg_signals)], axis=1)

# Round all aggregated signals to discrete values: -1, 0, or +1
agg_cols = ["equal_weight", "accuracy_weight", "majority_vote", "xgboost_model"]
for col in agg_cols:
    if col in test_signal_df.columns:
        test_signal_df[col] = test_signal_df[col].apply(
            lambda x: int(round(x)) if pd.notna(x) else x
        )


#
# # XGBOOST
# # Generate labels from future price movement
# diff = train_df["price"].diff().shift(-1).fillna(0)
# threshold_val = train_df["price"] * threshold
# train_labels = diff.apply(lambda x: 1 if x > threshold_val.mean() else -1 if x < -threshold_val.mean() else 0)
#
# # Align labels and features
# X = train_signal_df.dropna()
# y = train_labels.loc[X.index]
#
# # Train XGBoost model
# xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=3)
# xgb_model.fit(X, y)
#
# # Save the model + feature order
# model_bundle = {"model": xgb_model, "features": list(X.columns)}
# with open("../data/xgboost_model.pkl", "wb") as f:
#     pickle.dump(model_bundle, f)
#
# print("✅ Trained and saved XGBoost model to ../data/xgboost_model.pkl")
#

end_time = time.time()
print(f"\n✅ Signal generation + aggregation completed in {end_time - start_time:.2f} seconds")

# Print optimized parameters
print("Optimized parameters per signal:")
print(json.dumps(opt_params, indent=2))

# Save test predictions with metadata
test_metadata = test_df[["timestamp", "price", "volume"]].reset_index(drop=True)
output_df = pd.concat([test_metadata, test_signal_df.reset_index(drop=True)], axis=1)
output_df.to_csv("../data/backtest_signal_results_agg.csv", index=False)
print("✅ Results saved to ../data/backtest_signal_results_agg.csv")

# Compute train/test accuracy using threshold
train_backtester = SignalBacktester(train_signal_df, train_df["price"], threshold=threshold)
test_backtester = SignalBacktester(test_signal_df, test_df["price"], threshold=threshold)

train_accuracy = train_backtester.accuracy()
test_accuracy = test_backtester.accuracy()

# Combine results
acc_df = pd.DataFrame({
    "Signal": list(test_accuracy.keys()),
    "Train Accuracy (%)": [round(train_accuracy[k] * 100, 2) for k in test_accuracy.keys()],
    "Test Accuracy (%)": [round(test_accuracy[k] * 100, 2) for k in test_accuracy.keys()],
})

# Save accuracy
acc_df.to_csv("../data/backtest_signal_accuracy_agg.csv", index=False)
print("\n✅ Train/Test Accuracy saved to ../data/backtest_signal_accuracy_agg.csv")
print(acc_df)

# Flatten and save best signal params
clean_params = []
for signal_name, params in opt_params.items():
    row = {"Indicator": signal_name}
    row.update(params)
    clean_params.append(row)

param_df = pd.DataFrame(clean_params)
param_df.to_csv("../data/best_signal_params.csv", index=False)
print("\n✅ Saved best_signal_params.csv")
