import os
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import argparse # For command-line arguments

# --- Configuration ---
# SYMBOLS_TO_DOWNLOAD = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"] # Default list
DATA_TYPE = "trades"
MARKET_TYPE = "futures/um" # Assuming futures, adjust if spot is needed for some pairs
BASE_URL_TEMPLATE = "https://data.binance.vision/data/{market_type}/daily/{data_type}/{symbol}"

OUTPUT_DATA_DIR = "data"
DOWNLOAD_TEMP_DIR = os.path.join(OUTPUT_DATA_DIR, "temp_downloads")

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_TEMP_DIR, exist_ok=True)

DEFAULT_DAYS_TO_DOWNLOAD = 2 # For combined files, or for single "previous_day_N" files
# Resampling options
DEFAULT_RESAMPLE_INTERVAL = "1Min" # Changed to 1Min as it's more common for pairs trading analysis

def download_and_extract(symbol, date_str, market_type, data_type):
    base_url = BASE_URL_TEMPLATE.format(market_type=market_type, data_type=data_type, symbol=symbol)
    zip_filename = f"{symbol}-{data_type}-{date_str}.zip"
    csv_filename = f"{symbol}-{data_type}-{date_str}.csv" # Target CSV name after extraction
    
    zip_url = f"{base_url}/{zip_filename}"
    zip_path = os.path.join(DOWNLOAD_TEMP_DIR, zip_filename)
    extracted_csv_target_path = os.path.join(DOWNLOAD_TEMP_DIR, csv_filename)

    print(f"  Attempting to download for {symbol}: {zip_url}")
    try:
        resp = requests.get(zip_url, timeout=45)
        if resp.status_code == 200:
            with open(zip_path, "wb") as f:
                f.write(resp.content)
            # print(f"    ✅ Downloaded {zip_filename}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_files_in_zip = [name for name in zip_ref.namelist() if name.lower().endswith('.csv')]
                if not csv_files_in_zip:
                    print(f"    ❌ No CSV file found in {zip_filename} for {symbol}")
                    if os.path.exists(zip_path): os.remove(zip_path)
                    return None
                
                actual_csv_in_zip = csv_files_in_zip[0]
                extracted_file_original_path = os.path.join(DOWNLOAD_TEMP_DIR, actual_csv_in_zip)
                zip_ref.extract(actual_csv_in_zip, DOWNLOAD_TEMP_DIR)
                
                if extracted_file_original_path != extracted_csv_target_path:
                    if os.path.exists(extracted_csv_target_path): os.remove(extracted_csv_target_path)
                    os.rename(extracted_file_original_path, extracted_csv_target_path)
                
                # print(f"    ✅ Extracted to {extracted_csv_target_path}")
                return extracted_csv_target_path
        elif resp.status_code == 404:
            # print(f"    ℹ️  Data for {symbol} on {date_str} not found (HTTP 404).")
            pass # Less verbose for 404s
        else:
            print(f"    ❌ Failed to download {zip_filename} for {symbol} (HTTP {resp.status_code})")
        return None
    except requests.exceptions.RequestException as e:
        print(f"    ❌ Request error for {symbol} {zip_filename}: {e}")
    except zipfile.BadZipFile:
        print(f"    ❌ BadZipFile for {symbol} {zip_filename}.")
        if os.path.exists(zip_path): os.remove(zip_path)
    except Exception as e:
        print(f"    ❌ Unexpected error for {symbol} {zip_filename}: {e}")
    return None

def process_and_save_symbol_data(symbol_df, symbol, output_filename_base, resample_interval):
    if symbol_df.empty:
        print(f"  Input DataFrame for {symbol} is empty. Skipping save.")
        return

    df_processed = symbol_df.copy()
    # Ensure 'timestamp' is datetime for sorting and resampling
    df_processed["datetime_timestamp"] = pd.to_datetime(df_processed["timestamp_ms"], unit='ms')
    df_processed.sort_values("datetime_timestamp", inplace=True)
    df_processed.drop_duplicates(subset=['datetime_timestamp', 'price', 'volume'], inplace=True) # Remove exact duplicates
    df_processed.reset_index(drop=True, inplace=True)

    # Save the raw data (using original ms timestamp for 'time' column)
    output_df_raw = df_processed[['timestamp_ms', 'price', 'volume']].copy()
    output_df_raw.rename(columns={'timestamp_ms': 'time'}, inplace=True) # 'time' (ms)

    raw_output_path = os.path.join(OUTPUT_DATA_DIR, f"{symbol}-{output_filename_base}.csv")
    output_df_raw.to_csv(raw_output_path, index=False)
    print(f"  ✅ Raw {symbol} data saved to: {raw_output_path} ({len(output_df_raw)} rows)")

    if resample_interval:
        resampled_output_path = os.path.join(OUTPUT_DATA_DIR, f"{symbol}-{output_filename_base}-resampled-{resample_interval}.csv")
        # print(f"  Resampling {symbol} data to {resample_interval}...")
        try:
            df_to_resample = df_processed.set_index("datetime_timestamp")
            # For pairs trading, typically 'close' or 'last' price of the interval is used.
            # VWAP can also be good if volume data is reliable.
            agg_methods = {'price': 'last', 'volume': 'sum'} 
            resampled_df = df_to_resample.resample(resample_interval).agg(agg_methods)
            
            resampled_df['price'] = resampled_df['price'].ffill() # Forward fill price for missing intervals
            resampled_df['volume'] = resampled_df['volume'].fillna(0) # Fill volume with 0
            resampled_df.dropna(subset=['price'], inplace=True)
            resampled_df.reset_index(inplace=True) # 'datetime_timestamp' becomes a column
            
            if not resampled_df.empty:
                resampled_df.rename(columns={'datetime_timestamp': 'time_dt'}, inplace=True)
                resampled_df['time'] = (resampled_df['time_dt'].astype(np.int64) // 10**6) # ms
                resampled_df_for_csv = resampled_df[['time', 'price', 'volume']]
                resampled_df_for_csv.to_csv(resampled_output_path, index=False)
                print(f"  ✅ Resampled {symbol} data saved to: {resampled_output_path} ({len(resampled_df_for_csv)} rows)")
            else:
                print(f"  ℹ️  Resampled {symbol} data is empty. Not saved.")
        except Exception as e:
            print(f"  ❌ Error during resampling for {symbol}: {e}")

def main(symbols_to_download, days_to_fetch, resample_interval):
    print(f"Starting data download for symbols: {', '.join(symbols_to_download)}")
    print(f"Fetching data for the last {days_to_fetch} day(s). Resampling interval: {resample_interval or 'None'}")

    for symbol in symbols_to_download:
        print(f"\nProcessing symbol: {symbol}")
        daily_dfs_for_symbol = []
        download_success_count = 0

        for i in range(1, days_to_fetch + 2): # Try an extra day as buffer
            if download_success_count >= days_to_fetch: # Got enough days for this symbol
                break
            target_date = datetime.now(timezone.utc) - timedelta(days=i)
            date_str = target_date.strftime("%Y-%m-%d")
            
            csv_path = download_and_extract(symbol, date_str, MARKET_TYPE, DATA_TYPE)
            if csv_path:
                try:
                    df = pd.read_csv(csv_path)
                    # Standardize column names (Binance 'time' is ms epoch)
                    df.rename(columns={"price": "price", "qty": "volume", "time": "timestamp_ms"}, inplace=True)
                    required_cols = ["price", "volume", "timestamp_ms"]
                    if not all(col in df.columns for col in required_cols):
                        print(f"    ⚠️ File {csv_path} for {symbol} missing columns. Skipping.")
                        continue
                    daily_dfs_for_symbol.append(df[required_cols])
                    download_success_count += 1
                except Exception as e:
                    print(f"    ❌ Error reading/processing CSV {csv_path} for {symbol}: {e}")
            time.sleep(0.2) # Polite delay

        if not daily_dfs_for_symbol:
            print(f"  No data successfully downloaded/processed for {symbol}.")
            continue

        # Concatenate all downloaded daily DataFrames for this symbol
        combined_df_for_symbol = pd.concat(daily_dfs_for_symbol, ignore_index=True)
        
        # Determine output filename base (e.g., "trades-latest-2days")
        filename_suffix = f"{DATA_TYPE}-latest-{days_to_fetch}days" if days_to_fetch > 1 else f"{DATA_TYPE}-previous-day"
        process_and_save_symbol_data(combined_df_for_symbol, symbol, filename_suffix, resample_interval)

    print("\nCleaning up temporary download files...")
    for item in os.listdir(DOWNLOAD_TEMP_DIR):
        item_path = os.path.join(DOWNLOAD_TEMP_DIR, item)
        try:
            if os.path.isfile(item_path): os.unlink(item_path)
        except Exception as e:
            print(f'  Failed to delete {item_path}. Reason: {e}')
    print("\nData download and processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance trading data for specified symbols.")
    parser.add_argument(
        "-s", "--symbols",
        nargs='+',
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"],
        help="List of symbols to download (e.g., BTCUSDT ETHUSDT)."
    )
    parser.add_argument(
        "-d", "--days",
        type=int,
        default=DEFAULT_DAYS_TO_DOWNLOAD,
        help=f"Number of past days to download and combine for each symbol (default: {DEFAULT_DAYS_TO_DOWNLOAD}). Use 1 for 'previous-day'."
    )
    parser.add_argument(
        "-r", "--resample",
        type=str,
        default=DEFAULT_RESAMPLE_INTERVAL,
        help=f"Resampling interval (e.g., 1Min, 5S, 1H, or None to skip. Default: {DEFAULT_RESAMPLE_INTERVAL})."
    )
    args = parser.parse_args()

    resample_arg = args.resample if args.resample and args.resample.lower() != 'none' else None
    
    main(args.symbols, args.days, resample_arg)

"""
HOW TO USE:

# Download 2 days of data for default symbols, resample to 1Min
python download_binance_data.py

# Download 1 day (previous day) of data for BTC and ETH, no resampling
python download_binance_data.py --symbols BTCUSDT ETHUSDT --days 1 --resample None

# Download 5 days of data for SOL, resample to 5Min
python download_binance_data.py --symbols SOLUSDT --days 5 --resample 5Min
```This will create files like:
*   `data/BTCUSDT-trades-latest-2days.csv`
*   `data/BTCUSDT-trades-latest-2days-resampled-1Min.csv`
*   `data/ETHUSDT-trades-latest-2days.csv`
*   etc.

"""