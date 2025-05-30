import os
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta

# Binance link and file directory
symbol = "BTCUSDT"
base_url = f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}"
download_dir = "binance_trades_2days"
os.makedirs(download_dir, exist_ok=True)

# Output files naming & to set the resample interval
output_file = "merged_btcusdt_trades.csv"
resample_interval = "0.5s"  # Change to "2s", "5s"
resampled_file = f"btcusdt_{resample_interval}_resample.csv"

# Get the dates -> yesterday and the day before
dates = [(datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 3)]

# Download the data from binance
csv_files = []
for date_str in dates:
    zip_name = f"{symbol}-trades-{date_str}.zip"
    zip_url = f"{base_url}/{zip_name}"
    zip_path = os.path.join(download_dir, zip_name)

    print(f"Downloading {zip_url}...")
    resp = requests.get(zip_url)
    if resp.status_code != 200:
        print(f"❌ Failed to download {zip_url} (HTTP {resp.status_code})")
        continue

    with open(zip_path, "wb") as f:
        f.write(resp.content)
    print(f"✅ Downloaded {zip_name}")

    # Extract CSV
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
        extracted_files = zip_ref.namelist()
        csv_file_path = os.path.join(download_dir, extracted_files[0])
        csv_files.append(csv_file_path)

# Data cleaning
df_list = []
for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    df.rename(columns={
        "id": "trade_id",
        "price": "price",
        "qty": "quantity",
        "quoteQty": "quote_quantity",
        "time": "timestamp",
        "isBuyerMaker": "is_buyer_maker"
    }, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df_list.append(df)

# Merge data and sort by timestamp (ascending)
if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.sort_values("timestamp", inplace=True)
    merged_df.to_csv(output_file, index=False)
    print(f"\n✅ Merged csv saved to: {output_file}")

    # Resample data : can change the input (0.5s/1s/2s/5s/etc)
    df_resampled = merged_df.set_index("timestamp").resample(resample_interval).agg({
        "price": "last", # based on last trade price
        "quantity": "sum", # sum the quantity
        "is_buyer_maker": "last"
    }).dropna().reset_index()

    df_resampled.rename(columns={"quantity": "volume"}, inplace=True)
    df_resampled.to_csv(resampled_file, index=False)
    print(f"✅ Resampled csv saved to: {resampled_file}")
else:
    print("❌ No data to merge. Please check file availability.")
