import pandas as pd

# Load and parse
df = pd.read_csv('best_params_trade_log.csv')
df['time'] = pd.to_datetime(df['time'], dayfirst=True, infer_datetime_format=True)

# Pivot signals to wide format (last value per timestamp if duplicates)
signal_wide = df.pivot_table(
    index='time',
    columns='strategy',
    values='Signal',
    aggfunc='last'
).reset_index()

# Preview the result
print(signal_wide.head())

# Optionally save
signal_wide.to_csv('result/signals_rf_wide.csv', index=False)
print("Wide-format signals saved to 'result/signals_wide.csv'")