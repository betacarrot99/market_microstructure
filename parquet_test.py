import pandas as pd
import numpy as np
import time
import os
from pathlib import Path

def convert_csv_to_parquet(csv_file_path, parquet_file_path=None, optimize_dtypes=True):
    """
    Convert CSV to Parquet with optional data type optimization
    """
    if parquet_file_path is None:
        parquet_file_path = csv_file_path.replace('.csv', '.parquet')
    
    print(f"Converting {csv_file_path} to {parquet_file_path}...")
    
    # Load CSV
    start_time = time.time()
    df = pd.read_csv(csv_file_path)
    csv_load_time = time.time() - start_time
    
    if optimize_dtypes:
        print("Optimizing data types...")
        df = optimize_dataframe_dtypes(df)
    
    # Convert time column if it exists
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='ms')
    
    # Save as Parquet
    start_time = time.time()
    df.to_parquet(parquet_file_path, compression='snappy', index=False)
    parquet_save_time = time.time() - start_time
    
    # Get file sizes
    csv_size = os.path.getsize(csv_file_path) / (1024*1024)  # MB
    parquet_size = os.path.getsize(parquet_file_path) / (1024*1024)  # MB
    
    print(f"✅ Conversion complete!")
    print(f"📊 CSV load time: {csv_load_time:.3f}s")
    print(f"💾 Parquet save time: {parquet_save_time:.3f}s")
    print(f"📁 CSV size: {csv_size:.2f} MB")
    print(f"📁 Parquet size: {parquet_size:.2f} MB")
    print(f"🗜️ Size reduction: {((csv_size - parquet_size) / csv_size * 100):.1f}%")
    
    return parquet_file_path

def optimize_dataframe_dtypes(df):
    """
    Optimize DataFrame data types to reduce memory usage
    """
    original_memory = df.memory_usage(deep=True).sum() / (1024*1024)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to category if few unique values
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            # Downcast float64 to float32 if possible
            if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            # Downcast integers
            if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype('int32')
            elif df[col].min() >= np.iinfo(np.int16).min and df[col].max() <= np.iinfo(np.int16).max:
                df[col] = df[col].astype('int16')
    
    optimized_memory = df.memory_usage(deep=True).sum() / (1024*1024)
    print(f"🧠 Memory optimization: {original_memory:.2f} MB → {optimized_memory:.2f} MB ({((original_memory - optimized_memory) / original_memory * 100):.1f}% reduction)")
    
    return df

def benchmark_loading_performance(csv_file, parquet_file, num_runs=5):
    """
    Benchmark loading performance between CSV and Parquet
    """
    print(f"\n🏃‍♂️ Benchmarking loading performance ({num_runs} runs each)...")
    
    csv_times = []
    parquet_times = []
    
    # Benchmark CSV loading
    for i in range(num_runs):
        start_time = time.time()
        df_csv = pd.read_csv(csv_file)
        if 'time' in df_csv.columns:
            df_csv['time'] = pd.to_datetime(df_csv['time'], unit='ms')
        csv_times.append(time.time() - start_time)
    
    # Benchmark Parquet loading
    for i in range(num_runs):
        start_time = time.time()
        df_parquet = pd.read_parquet(parquet_file)
        parquet_times.append(time.time() - start_time)
    
    csv_avg = np.mean(csv_times)
    parquet_avg = np.mean(parquet_times)
    speedup = csv_avg / parquet_avg
    
    print(f"📊 Results:")
    print(f"   CSV average load time: {csv_avg:.3f}s (±{np.std(csv_times):.3f}s)")
    print(f"   Parquet average load time: {parquet_avg:.3f}s (±{np.std(parquet_times):.3f}s)")
    print(f"   🚀 Speedup: {speedup:.1f}x faster with Parquet!")
    
    return csv_avg, parquet_avg, speedup

def batch_convert_directory(directory_path, file_pattern="*.csv"):
    """
    Convert all CSV files in a directory to Parquet
    """
    directory = Path(directory_path)
    csv_files = list(directory.glob(file_pattern))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files to convert...")
    
    total_csv_size = 0
    total_parquet_size = 0
    total_speedup = []
    
    for csv_file in csv_files:
        print(f"\n📁 Processing: {csv_file.name}")
        
        # Convert to parquet
        parquet_file = csv_file.with_suffix('.parquet')
        convert_csv_to_parquet(str(csv_file), str(parquet_file))
        
        # Update totals
        total_csv_size += os.path.getsize(csv_file) / (1024*1024)
        total_parquet_size += os.path.getsize(parquet_file) / (1024*1024)
        
        # Quick benchmark
        csv_time = []
        parquet_time = []
        for _ in range(3):
            start = time.time()
            pd.read_csv(csv_file)
            csv_time.append(time.time() - start)
            
            start = time.time()
            pd.read_parquet(parquet_file)
            parquet_time.append(time.time() - start)
        
        speedup = np.mean(csv_time) / np.mean(parquet_time)
        total_speedup.append(speedup)
        print(f"   ⚡ Load speedup: {speedup:.1f}x")
    
    print(f"\n🎉 Batch conversion complete!")
    print(f"📊 Summary:")
    print(f"   Total CSV size: {total_csv_size:.2f} MB")
    print(f"   Total Parquet size: {total_parquet_size:.2f} MB")
    print(f"   Overall size reduction: {((total_csv_size - total_parquet_size) / total_csv_size * 100):.1f}%")
    print(f"   Average speedup: {np.mean(total_speedup):.1f}x")

# Updated trading strategy functions to use Parquet
def load_trading_data_optimized(file_path, use_parquet=True):
    """
    Optimized data loading function
    """
    if use_parquet and file_path.endswith('.csv'):
        parquet_path = file_path.replace('.csv', '.parquet')
        
        if os.path.exists(parquet_path):
            print(f"📈 Loading Parquet file: {parquet_path}")
            start_time = time.time()
            data = pd.read_parquet(parquet_path)
            load_time = time.time() - start_time
            print(f"⚡ Loaded in {load_time:.3f}s")
        else:
            print(f"🔄 Parquet file not found, converting from CSV...")
            convert_csv_to_parquet(file_path, parquet_path)
            data = pd.read_parquet(parquet_path)
    else:
        print(f"📊 Loading CSV file: {file_path}")
        start_time = time.time()
        data = pd.read_csv(file_path)
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'], unit='ms')
        load_time = time.time() - start_time
        print(f"⏱️ Loaded in {load_time:.3f}s")
    
    # Rename price column if needed
    if 'price' in data.columns:
        data = data.rename(columns={'price': 'last_trade_price'})
    
    return data

# Example usage and testing
if __name__ == "__main__":
    # Example: Convert your BTCUSDT data
    csv_file = 'data/BTCUSDT-trades-2025-05-20.csv'
    
    # Check if file exists (adjust path as needed)
    if os.path.exists(csv_file):
        print("🚀 Starting CSV to Parquet conversion...")
        
        # Convert single file
        parquet_file = convert_csv_to_parquet(csv_file)
        
        # Benchmark performance
        benchmark_loading_performance(csv_file, parquet_file)
        
        print("\n" + "="*60)
        print("📋 INTEGRATION TIPS:")
        print("="*60)
        print("1. Replace your pd.read_csv() calls with:")
        print("   data = load_trading_data_optimized('data/BTCUSDT-trades-2025-05-20.csv')")
        print("\n2. For batch conversion of all CSV files:")
        print("   batch_convert_directory('data/')")
        print("\n3. Parquet files are typically 50-80% smaller")
        print("4. Loading is typically 3-10x faster")
        print("5. Preserves data types automatically")
        print("="*60)
        
    else:
        print(f"❌ File not found: {csv_file}")
        print("Please update the file path to match your data location")
        
        # Show example for batch conversion
        print("\n📁 For batch conversion, use:")
        print("batch_convert_directory('your_data_directory')")

# Integration example for your existing code
def updated_ewma_strategy_example():
    """
    Example of how to integrate Parquet loading into your existing strategy
    """
    import time
    
    start_time = time.time()
    
    # Updated data loading (automatically uses Parquet if available)
    data = load_trading_data_optimized('data/BTCUSDT-trades-2025-05-20.csv')
    
    # Rest of your EWMA strategy code remains the same
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"✅ Data loaded and prepared in {time.time() - start_time:.3f}s")
    print(f"📊 Dataset shape: {data.shape}")
    print(f"🗓️ Date range: {data['time'].min()} to {data['time'].max()}")
    
    return data

# Show the integration example
print("\n" + "="*60)
print("🔧 INTEGRATION EXAMPLE")
print("="*60)
print("Replace this in your existing code:")
print("   data2 = pd.read_csv('data/BTCUSDT-trades-2025-05-20.csv')")
print("   data['time'] = pd.to_datetime(data['time'], unit='ms')")
print("   data = data.rename(columns={'price': 'last_trade_price'})")
print("\nWith this:")
print("   data2 = load_trading_data_optimized('data/BTCUSDT-trades-2025-05-20.csv')")
print("="*60)