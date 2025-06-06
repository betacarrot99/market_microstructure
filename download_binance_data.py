import os
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone # Added timezone
import time

# --- Configuration ---
SYMBOL = "BTCUSDT"
DATA_TYPE = "trades"
MARKET_TYPE = "futures/um"
BASE_URL = f"https://data.binance.vision/data/{MARKET_TYPE}/daily/{DATA_TYPE}/{SYMBOL}"

OUTPUT_DATA_DIR = "data"
DOWNLOAD_TEMP_DIR = os.path.join(OUTPUT_DATA_DIR, "temp_downloads")

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_TEMP_DIR, exist_ok=True)

# --- Output file naming ---
# For 2-day combined data
FINAL_MERGED_2DAYS_FILE = os.path.join(OUTPUT_DATA_DIR, f"{SYMBOL}-{DATA_TYPE}-latest-2days.csv")
RESAMPLED_2DAYS_FILE = None # Initialize

# For single previous day data
FINAL_PREVIOUS_DAY_FILE = os.path.join(OUTPUT_DATA_DIR, f"{SYMBOL}-{DATA_TYPE}-previous-day.csv")
RESAMPLED_PREVIOUS_DAY_FILE = None # Initialize

# Resampling options (set to None to skip resampling globally, or configure per output)
RESAMPLE_INTERVAL = "100ms" # e.g., "1S", "1Min", "5Min", "100ms", or None

if RESAMPLE_INTERVAL:
    RESAMPLED_2DAYS_FILE = os.path.join(OUTPUT_DATA_DIR, f"{SYMBOL}-{DATA_TYPE}-latest-2days-resampled-{RESAMPLE_INTERVAL}.csv")
    RESAMPLED_PREVIOUS_DAY_FILE = os.path.join(OUTPUT_DATA_DIR, f"{SYMBOL}-{DATA_TYPE}-previous-day-resampled-{RESAMPLE_INTERVAL}.csv")

DAYS_TO_DOWNLOAD_FOR_2DAY_FILE = 2 # Number of most recent daily files to combine for the 2-day dataset

# --- Helper Functions ---
def download_and_extract(date_str):
    """Downloads and extracts data for a specific date string (YYYY-MM-DD)."""
    zip_filename = f"{SYMBOL}-{DATA_TYPE}-{date_str}.zip"
    csv_filename = f"{SYMBOL}-{DATA_TYPE}-{date_str}.csv"
    
    zip_url = f"{BASE_URL}/{zip_filename}"
    zip_path = os.path.join(DOWNLOAD_TEMP_DIR, zip_filename)
    extracted_csv_path = os.path.join(DOWNLOAD_TEMP_DIR, csv_filename) # Target name

    print(f"Attempting to download: {zip_url}")
    try:
        resp = requests.get(zip_url, timeout=30)
        if resp.status_code == 200:
            with open(zip_path, "wb") as f:
                f.write(resp.content)
            print(f"  ✅ Downloaded {zip_filename}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_files_in_zip = [name for name in zip_ref.namelist() if name.lower().endswith('.csv')]
                if not csv_files_in_zip:
                    print(f"  ❌ No CSV file found in {zip_filename}")
                    if os.path.exists(zip_path): os.remove(zip_path)
                    return None
                
                actual_csv_in_zip = csv_files_in_zip[0] # Assume first CSV is the one
                # Full path to the actually extracted file (might have a different case or subfolder in zip)
                current_extracted_file_path_in_temp = os.path.join(DOWNLOAD_TEMP_DIR, actual_csv_in_zip)
                zip_ref.extract(actual_csv_in_zip, DOWNLOAD_TEMP_DIR)
                
                # Rename to our standard expected name if different
                if current_extracted_file_path_in_temp != extracted_csv_path:
                    # Ensure target does not exist if renaming (e.g. due to case differences on some OS)
                    if os.path.exists(extracted_csv_path) and current_extracted_file_path_in_temp.lower() == extracted_csv_path.lower() :
                        os.remove(extracted_csv_path) # remove if it's just a case difference issue
                    os.rename(current_extracted_file_path_in_temp, extracted_csv_path)
                
                print(f"  ✅ Extracted to {extracted_csv_path}")
                return extracted_csv_path

        elif resp.status_code == 404:
            print(f"  ℹ️  Data for {date_str} not found (HTTP 404).")
            return None
        else:
            print(f"  ❌ Failed to download {zip_filename} (HTTP {resp.status_code})")
            return None
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Request error downloading {zip_filename}: {e}")
    except zipfile.BadZipFile:
        print(f"  ❌ BadZipFile error for {zip_filename}. Download may be incomplete or corrupted.")
        if os.path.exists(zip_path): os.remove(zip_path)
    except Exception as e:
        print(f"  ❌ An unexpected error occurred with {zip_filename}: {e}")
    return None


def process_and_save_dataframe(df_input, base_output_filename, resampled_output_filename, description):
    """Processes a DataFrame and saves raw and optionally resampled versions."""
    if df_input.empty:
        print(f"Input DataFrame for {description} is empty. Skipping save.")
        return

    # Create a 'datetime_timestamp' column for processing, keep original 'timestamp' (ms)
    # The input df_input should already have 'timestamp' (ms), 'price', 'volume'
    df_processed = df_input.copy()
    df_processed["datetime_timestamp"] = pd.to_datetime(df_processed["timestamp"], unit='ms')
    df_processed.sort_values("datetime_timestamp", inplace=True)
    df_processed.reset_index(drop=True, inplace=True)

    # Save the raw data (with 'time' in ms, 'price', 'volume')
    # df_for_clustering expects 'time' (ms), 'price', 'volume'
    output_df_raw = df_processed[['timestamp', 'price', 'volume']].copy()
    output_df_raw.rename(columns={'timestamp': 'time'}, inplace=True) # Ensure 'time' column for clustering.py

    output_df_raw.to_csv(base_output_filename, index=False)
    print(f"\n✅ {description} data saved to: {base_output_filename}")
    print(f"   This file contains {len(output_df_raw)} trades.")
    if not output_df_raw.empty:
        min_dt = pd.to_datetime(output_df_raw['time'].min(), unit='ms', utc=True)
        max_dt = pd.to_datetime(output_df_raw['time'].max(), unit='ms', utc=True)
        print(f"   Date range: {min_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} to {max_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Optional: Resample data
    if RESAMPLE_INTERVAL and resampled_output_filename:
        print(f"Resampling {description} data to {RESAMPLE_INTERVAL} interval...")
        if 'datetime_timestamp' not in df_processed.columns or df_processed['datetime_timestamp'].isnull().all():
            print("  ⚠️ Cannot resample, 'datetime_timestamp' column is missing or all NaNs.")
            return
        try:
            df_to_resample = df_processed.set_index("datetime_timestamp")
            agg_methods_simple = {'price': 'last', 'volume': 'sum'}
            resampled_df = df_to_resample.resample(RESAMPLE_INTERVAL).agg(agg_methods_simple)
            
            resampled_df['price'] = resampled_df['price'].ffill()
            resampled_df['volume'] = resampled_df['volume'].fillna(0)
            resampled_df.dropna(subset=['price'], inplace=True) # Crucial: drop rows where price is still NaN
            resampled_df.reset_index(inplace=True)
            
            # Rename 'datetime_timestamp' back to 'time' and convert to ms
            resampled_df.rename(columns={'datetime_timestamp': 'time_dt'}, inplace=True) # temp rename
            resampled_df['time'] = (resampled_df['time_dt'].astype(np.int64) // 10**6)
            resampled_df_for_csv = resampled_df[['time', 'price', 'volume']]

            if not resampled_df_for_csv.empty:
                resampled_df_for_csv.to_csv(resampled_output_filename, index=False)
                print(f"  ✅ Resampled {description} data saved to: {resampled_output_filename}")
                print(f"     Resampled data contains {len(resampled_df_for_csv)} rows.")
            else:
                print(f"  ℹ️  Resampled {description} data is empty after processing. Not saved.")
        except Exception as e:
            print(f"  ❌ Error during resampling for {description}: {e}")

def main():
    print(f"Starting data download for {SYMBOL}...")
    
    all_downloaded_csv_paths = [] # Store paths of all successfully downloaded daily files
    # Try to download enough days for the 2-day file, plus a small buffer
    # We iterate from yesterday backwards.
    for i in range(1, DAYS_TO_DOWNLOAD_FOR_2DAY_FILE + 3): # e.g., for 2 days, try up to 4 days ago
        # Update to use datetime.now(timezone.utc)
        target_date = datetime.now(timezone.utc) - timedelta(days=i)
        date_str = target_date.strftime("%Y-%m-%d")
        
        csv_path = download_and_extract(date_str)
        if csv_path:
            all_downloaded_csv_paths.append({"date_str": date_str, "path": csv_path})
        
        if len(all_downloaded_csv_paths) >= DAYS_TO_DOWNLOAD_FOR_2DAY_FILE: # Ensure we have at least enough for 2-day
            # We can break early if we only care about a strict number of files,
            # but downloading an extra one can be a good fallback if the most recent is partial.
            # For this logic, we'll continue up to the loop limit to get the freshest possible set.
            pass 
        time.sleep(0.5) 

    if not all_downloaded_csv_paths:
        print("No data files were successfully downloaded. Exiting.")
        return

    # Sort downloaded files by date string (newest first based on how we iterated)
    # No, we iterated i=1 (yesterday), i=2 (day before), so all_downloaded_csv_paths is already newest first.
    # If we want to be explicit:
    # all_downloaded_csv_paths.sort(key=lambda x: x['date_str'], reverse=True)

    print(f"\nSuccessfully downloaded {len(all_downloaded_csv_paths)} daily files.")

    # --- Process for 2-Day Combined File ---
    if len(all_downloaded_csv_paths) >= 1: # Need at least one file
        # Select the most recent files needed for the 2-day combined data
        files_for_2day_combined = [item['path'] for item in all_downloaded_csv_paths[:DAYS_TO_DOWNLOAD_FOR_2DAY_FILE]]
        print(f"\nProcessing {len(files_for_2day_combined)} files for the 'latest-2days' dataset: {files_for_2day_combined}")
        
        df_list_2days = []
        for file_path in files_for_2day_combined:
            try:
                df = pd.read_csv(file_path)
                df.rename(columns={"price": "price", "qty": "volume", "time": "timestamp"}, inplace=True)
                required_cols = ["price", "volume", "timestamp"]
                if not all(col in df.columns for col in required_cols):
                    print(f"  ⚠️  File {file_path} (for 2-day) missing columns. Skipping.")
                    continue
                df = df[required_cols]
                # 'timestamp' column is already in milliseconds from Binance CSV
                df_list_2days.append(df)
                print(f"  ✅ Read {os.path.basename(file_path)} for 2-day merge")
            except Exception as e:
                print(f"  ❌ Error processing {file_path} (for 2-day): {e}")

        if df_list_2days:
            merged_df_2days = pd.concat(df_list_2days, ignore_index=True)
            # No need to sort by timestamp yet, process_and_save_dataframe will do it
            process_and_save_dataframe(merged_df_2days, FINAL_MERGED_2DAYS_FILE, RESAMPLED_2DAYS_FILE, "Latest 2-Days Combined")
        else:
            print("Could not prepare data for the 'latest-2days' file.")
    else:
        print("Not enough daily files downloaded to create 'latest-2days' dataset.")


    # --- Process for Single Previous Day File ---
    # The first file in all_downloaded_csv_paths is the most recent successfully downloaded daily file
    if all_downloaded_csv_paths:
        previous_day_file_info = all_downloaded_csv_paths[0] # This is yesterday's or the newest available
        print(f"\nProcessing file for 'previous-day' dataset: {previous_day_file_info['path']}")
        
        try:
            df_previous_day = pd.read_csv(previous_day_file_info['path'])
            df_previous_day.rename(columns={"price": "price", "qty": "volume", "time": "timestamp"}, inplace=True)
            required_cols = ["price", "volume", "timestamp"]
            if not all(col in df_previous_day.columns for col in required_cols):
                print(f"  ⚠️  File {previous_day_file_info['path']} (for previous-day) missing columns. Skipping previous day processing.")
            else:
                df_previous_day = df_previous_day[required_cols]
                # 'timestamp' is already in ms
                process_and_save_dataframe(df_previous_day, FINAL_PREVIOUS_DAY_FILE, RESAMPLED_PREVIOUS_DAY_FILE, "Previous Day")
        except Exception as e:
            print(f"  ❌ Error processing {previous_day_file_info['path']} (for previous-day): {e}")
    else:
        print("No files available to process for the 'previous-day' dataset.")

            
    print("\nCleaning up temporary download files...")
    for item in os.listdir(DOWNLOAD_TEMP_DIR):
        item_path = os.path.join(DOWNLOAD_TEMP_DIR, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
        except Exception as e:
            print(f'  Failed to delete {item_path}. Reason: {e}')
    print("\nData download and processing complete.")

if __name__ == "__main__":
    main()