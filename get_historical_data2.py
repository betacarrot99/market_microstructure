# get_historical_data.py
import os
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta

# Binance API config and file directory
symbol = "BTCUSDT"
base_url = f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}"
download_dir = "binance_trades_2days"
os.makedirs(download_dir, exist_ok=True)

# Output files naming & to set the resample interval
output_file = "data/merged_BTCUSDT_trades.csv"
resample_interval = "1s"  # Change to "10ms" or "0.1s" or "2min"
resampled_file = f"data/BTCUSDT_{resample_interval}_resample.csv"

# Download latest past 2 days trade data (at least 2 csv files)
csv_files = []
days_back = 1
while len(csv_files) < 2:
    date_str = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    zip_name = f"{symbol}-trades-{date_str}.zip"
    zip_url = f"{base_url}/{zip_name}"
    zip_path = os.path.join(download_dir, zip_name)

    print(f"Downloading: {zip_url}")
    resp = requests.get(zip_url)
    if resp.status_code == 200:
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        print(f"Downloaded {zip_name}")

        # Extract csv
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
            extracted_files = zip_ref.namelist()
            csv_file_path = os.path.join(download_dir, extracted_files[0])
            csv_files.append(csv_file_path)
    else:
        print(f" {date_str} not available (HTTP {resp.status_code})")

    days_back += 1

# Data cleaning
df_list = []
csv_files = ['binance_trades_2days/BTCUSDT-trades-2025-06-21.csv','binance_trades_2days/BTCUSDT-trades-2025-06-22.csv']
for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    df_list.append(df)

# Merge and sort
merged_df = pd.concat(df_list, ignore_index=True)
merged_df.sort_values("time", inplace=True)
merged_df.to_csv(output_file, index=False)
print(f"\nMerged CSV saved to: {output_file}")

# Resample
merged_df["time"] = pd.to_datetime(merged_df["time"], unit='ms')
df_resampled = merged_df.set_index("time").resample(resample_interval).agg({
    "id" : "last",
    "price": ["last", "max", "min"],
    "qty": "sum",
    # "quote_qty": "sum",
    # "is_buyer_maker": "last"
}).dropna().reset_index()

# df_resampled["time"] = (df_resampled["time"].astype("int64") // 1_000_000).astype("int64")

# Flatten multi-level columns after aggregation
df_resampled.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_resampled.columns]
df_resampled = df_resampled.reset_index()

# Rename price column
df_resampled.rename(columns={
    "time_": "time",
    "id_last": "trade_id",
    "price_last": "price",
    "price_max": "high",
    "price_min": "low",
    "qty_sum": "qty",
    # "quote_qty_sum": "quote_qty",
    # "is_buyer_maker_last": "is_buyer_maker"
}, inplace=True)

df_resampled.to_csv(resampled_file, index=False)
print(f"Resampled CSV saved to: {resampled_file}")