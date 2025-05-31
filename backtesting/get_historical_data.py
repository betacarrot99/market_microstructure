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
output_file = "merged_btcusdt_trades.csv"
resample_interval = "100ms"  # Change to "10ms" or "0.1s" or "2min"
resampled_file = f"btcusdt_{resample_interval}_resample.csv"

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
        print(f"✅ Downloaded {zip_name}")

        # Extract csv
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
            extracted_files = zip_ref.namelist()
            csv_file_path = os.path.join(download_dir, extracted_files[0])
            csv_files.append(csv_file_path)
    else:
        print(f"❌ {date_str} not available (HTTP {resp.status_code})")

    days_back += 1

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

# Merge and sort
merged_df = pd.concat(df_list, ignore_index=True)
merged_df.sort_values("timestamp", inplace=True)
merged_df.rename(columns={"quantity": "volume"}, inplace=True)
merged_df = merged_df[["timestamp", "price", "volume", "is_buyer_maker"]]
merged_df.to_csv(output_file, index=False)
print(f"\n✅ Merged CSV saved to: {output_file}")


# Resample
df_resampled = merged_df.set_index("timestamp").resample(resample_interval).agg({
    "price": "mean",
    "volume": "sum",
    "is_buyer_maker": "last"
}).dropna().reset_index()

df_resampled.to_csv(resampled_file, index=False)
print(f"✅ Resampled CSV saved to: {resampled_file}")
