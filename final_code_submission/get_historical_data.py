
import os
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta

# Binance API config and file directories
symbol = "BTCUSDT"
base_url = f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}"
download_dir = "binance_trades_2days"
data_dir = "data"
result_dir = "result"
os.makedirs(download_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)


# Output files and intervals
merged_file = os.path.join(data_dir, "merged_BTCUSDT_trades.csv")
resample_intervals = ["1min", "1s"]  # added 1-second interval

# Step 1: Download the last 2 days of trade data
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
        print(f"  → downloaded {zip_name}")

        # extract the single CSV inside
        with zipfile.ZipFile(zip_path, 'r') as z:
            inner_name = z.namelist()[0]
            z.extract(inner_name, download_dir)
            csv_files.append(os.path.join(download_dir, inner_name))
    else:
        print(f"  → {date_str} not available (HTTP {resp.status_code})")

    days_back += 1
# csv_files = ['binance_trades_2days/BTCUSDT-trades-2025-06-21.csv','binance_trades_2days/BTCUSDT-trades-2025-06-22.csv']
# Step 2: Merge & save raw trades
df_list = [pd.read_csv(f, low_memory=False) for f in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)
merged_df.sort_values("time", inplace=True)
merged_df.to_csv(merged_file, index=False)
print(f"\nMerged CSV saved to: {merged_file}")

# Step 3: Convert timestamp & set index
merged_df["time"] = pd.to_datetime(merged_df["time"], unit="ms")
merged_df.set_index("time", inplace=True)

# Step 4: Resample for each interval and save
for interval in resample_intervals:
    df_r = merged_df.resample(interval).agg({
        "id": "last",
        "price": ["last", "max", "min"],
        "qty": "sum",
    }).dropna().reset_index()

    # flatten MultiIndex columns, dropping the trailing underscore when level2 is empty
    df_r.columns = [
        f"{lvl0}_{lvl1}" if lvl1 else lvl0
        for lvl0, lvl1 in df_r.columns
    ]

    # rename to cleaner names
    df_r.rename(columns={
        "id_last":    "trade_id",
        "price_last": "price",
        "price_max":  "high",
        "price_min":  "low",
        "qty_sum":    "qty",
    }, inplace=True)

    out_path = os.path.join(data_dir, f"BTCUSDT_{interval}_resample.csv")
    df_r.to_csv(out_path, index=False)
    print(f"Resampled ({interval}) CSV saved to: {out_path}")
