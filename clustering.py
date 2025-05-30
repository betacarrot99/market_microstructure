# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# import os
# import time
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score
# import warnings
# warnings.filterwarnings('ignore')

# start_time = time.time()

# def create_technical_features(data, lookback_window=20):
#     """Create technical indicators for clustering"""
#     df = data.copy()
    
#     # Price-based features
#     df['returns'] = df['last_trade_price'].pct_change()
#     df['price_ma'] = df['last_trade_price'].rolling(window=lookback_window).mean()
#     df['price_std'] = df['last_trade_price'].rolling(window=lookback_window).std()
    
#     # Momentum indicators
#     df['rsi'] = calculate_rsi(df['last_trade_price'], lookback_window)
#     df['momentum'] = df['last_trade_price'] / df['last_trade_price'].shift(lookback_window) - 1
    
#     # Volatility indicators
#     df['volatility'] = df['returns'].rolling(window=lookback_window).std()
#     df['price_range'] = (df['last_trade_price'].rolling(window=lookback_window).max() - 
#                         df['last_trade_price'].rolling(window=lookback_window).min()) / df['last_trade_price']
    
#     # Price position indicators
#     df['price_percentile'] = df['last_trade_price'].rolling(window=lookback_window).rank(pct=True)
#     df['distance_from_ma'] = (df['last_trade_price'] - df['price_ma']) / df['price_ma']
    
#     return df

# def calculate_rsi(prices, window=14):
#     """Calculate Relative Strength Index"""
#     delta = prices.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi

# def clustering_strategy_with_tuning(data, cluster_range, lookback_range, feature_combinations):
#     """
#     Clustering-based trading strategy with parameter tuning
    
#     Parameters:
#     - cluster_range: range of cluster numbers to test
#     - lookback_range: range of lookback windows for technical indicators
#     - feature_combinations: list of feature sets to test
#     """
#     best_params = {
#         'n_clusters': None, 
#         'lookback_window': None, 
#         'features': None,
#         'accuracy': -np.inf
#     }
#     results = pd.DataFrame()
    
#     print(f"Testing {len(cluster_range)} cluster counts × {len(lookback_range)} lookback windows × {len(feature_combinations)} feature sets...")
    
#     total_combinations = len(cluster_range) * len(lookback_range) * len(feature_combinations)
#     current_combination = 0
    
#     for n_clusters in cluster_range:
#         for lookback_window in lookback_range:
#             for features in feature_combinations:
#                 current_combination += 1
#                 if current_combination % 10 == 0:
#                     print(f"Progress: {current_combination}/{total_combinations} ({current_combination/total_combinations*100:.1f}%)")
                
#                 df = create_technical_features(data, lookback_window)
                
#                 # Prepare feature matrix
#                 feature_cols = [col for col in features if col in df.columns]
#                 if len(feature_cols) < 2:  # Need at least 2 features for clustering
#                     continue
                
#                 # Remove rows with NaN values
#                 feature_data = df[feature_cols].dropna()
#                 if len(feature_data) < n_clusters * 10:  # Need enough data points
#                     continue
                
#                 # Standardize features
#                 scaler = StandardScaler()
#                 features_scaled = scaler.fit_transform(feature_data)
                
#                 # Perform clustering
#                 try:
#                     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#                     clusters = kmeans.fit_predict(features_scaled)
                    
#                     # Add cluster labels back to dataframe
#                     df_clustered = df.loc[feature_data.index].copy()
#                     df_clustered['cluster'] = clusters
                    
#                     # Generate signals based on cluster transitions
#                     df_clustered = generate_cluster_signals(df_clustered, n_clusters)
                    
#                     # Calculate market direction
#                     df_clustered['Market_Direction'] = np.where(
#                         df_clustered['last_trade_price'].shift(-1) > df_clustered['last_trade_price'], 1, 0
#                     )
                    
#                     # Calculate accuracy
#                     df_clustered['Correct'] = df_clustered['Signal'] == df_clustered['Market_Direction']
#                     accuracy = df_clustered['Correct'].mean()
                    
#                     # Update best parameters if this combination is better
#                     if accuracy > best_params['accuracy']:
#                         best_params = {
#                             'n_clusters': n_clusters,
#                             'lookback_window': lookback_window,
#                             'features': features,
#                             'accuracy': accuracy
#                         }
#                         results = df_clustered.copy()
                        
#                 except Exception as e:
#                     # Skip this combination if clustering fails
#                     continue
    
#     return best_params, results

# def generate_cluster_signals(df, n_clusters):
#     """
#     Generate trading signals based on cluster analysis
#     Multiple signal generation strategies
#     """
#     df = df.copy()
    
#     # Strategy 1: Cluster momentum - buy when moving to "bullish" clusters
#     cluster_returns = df.groupby('cluster')['returns'].mean()
#     bullish_clusters = cluster_returns.nlargest(n_clusters//2).index
    
#     df['cluster_momentum_signal'] = 0
#     for i in range(1, len(df)):
#         current_cluster = df.iloc[i]['cluster']
#         prev_cluster = df.iloc[i-1]['cluster']
        
#         # Signal generation based on cluster transitions
#         if current_cluster in bullish_clusters and prev_cluster not in bullish_clusters:
#             df.iloc[i, df.columns.get_loc('cluster_momentum_signal')] = 1
#         elif current_cluster not in bullish_clusters and prev_cluster in bullish_clusters:
#             df.iloc[i, df.columns.get_loc('cluster_momentum_signal')] = 0
#         else:
#             df.iloc[i, df.columns.get_loc('cluster_momentum_signal')] = df.iloc[i-1]['cluster_momentum_signal']
    
#     # Strategy 2: Cluster persistence - buy when in consistently bullish cluster
#     df['cluster_persistence_signal'] = df['cluster'].isin(bullish_clusters).astype(int)
    
#     # Strategy 3: Cluster volatility - buy in low volatility clusters during uptrends
#     cluster_volatility = df.groupby('cluster')['volatility'].mean()
#     low_vol_clusters = cluster_volatility.nsmallest(n_clusters//2).index
    
#     df['cluster_vol_signal'] = 0
#     df.loc[df['cluster'].isin(low_vol_clusters) & (df['returns'] > 0), 'cluster_vol_signal'] = 1
    
#     # Ensemble signal (majority vote)
#     signal_cols = ['cluster_momentum_signal', 'cluster_persistence_signal', 'cluster_vol_signal']
#     df['Signal'] = (df[signal_cols].sum(axis=1) >= 2).astype(int)
    
#     return df

# def log_best_params(file_path, timestamp, n_clusters, lookback_window, features, accuracy, 
#                    processing_time, input_start_date, input_end_date, signal_method):
#     """Log the best parameters to CSV"""
#     new_entry = pd.DataFrame([{
#         "timestamp": timestamp,
#         "n_clusters": n_clusters,
#         "lookback_window": lookback_window,
#         "features": str(features),
#         "accuracy": accuracy,
#         "processing_time": processing_time,
#         "input_date": input_start_date,
#         "input_end_date": input_end_date,
#         "method": signal_method
#     }])
    
#     if os.path.exists(file_path):
#         new_entry.to_csv(file_path, mode='a', index=False, header=False)
#     else:
#         new_entry.to_csv(file_path, index=False, header=True)

# # Main execution
# if __name__ == "__main__":
#     # Load and prepare data
#     data2 = pd.read_csv('data/BTCUSDT-trades-2025-05-20.csv')
#     # data1 = pd.read_csv('data/BTCUSDT-trades-2025-05-18.csv')
#     # data = pd.concat([data1, data2])
#     data = data2
#     data['time'] = pd.to_datetime(data['time'], unit='ms')
#     data = data.rename(columns={'price': 'last_trade_price'})
    
#     # Train-test split
#     train_size = int(0.8 * len(data))
#     train_data = data[:train_size]
#     test_data = data[train_size:]
    
#     # Define parameter ranges for tuning
#     cluster_range = range(3, 8)  # 3 to 7 clusters
#     lookback_range = range(10, 31, 5)  # 10, 15, 20, 25, 30
    
#     # Define different feature combinations to test
#     feature_combinations = [
#         ['returns', 'rsi', 'volatility', 'momentum'],
#         ['returns', 'price_percentile', 'distance_from_ma', 'volatility'],
#         ['rsi', 'momentum', 'price_range', 'distance_from_ma'],
#         ['returns', 'rsi', 'momentum', 'price_percentile', 'volatility'],
#         ['returns', 'volatility', 'price_range', 'distance_from_ma', 'momentum']
#     ]
    
#     print("Starting clustering strategy optimization...")
    
#     # Find best parameters on training data
#     best_params, train_results = clustering_strategy_with_tuning(
#         train_data,
#         cluster_range=cluster_range,
#         lookback_range=lookback_range,
#         feature_combinations=feature_combinations
#     )
    
#     print("\n" + "="*50)
#     print("Best Parameters (by accuracy):")
#     print(f"  Clusters: {best_params['n_clusters']}")
#     print(f"  Lookback Window: {best_params['lookback_window']}")
#     print(f"  Features: {best_params['features']}")
#     print(f"  Train signal accuracy: {best_params['accuracy']:.2%}")
#     print("="*50)
    
#     # Apply best parameters to test data
#     if best_params['n_clusters'] is not None:
#         # Recreate the model with best parameters
#         test_df = create_technical_features(test_data, best_params['lookback_window'])
        
#         # Prepare features
#         feature_cols = [col for col in best_params['features'] if col in test_df.columns]
#         feature_data = test_df[feature_cols].dropna()
        
#         if len(feature_data) > 0:
#             # Refit scaler and clustering model on training data features
#             train_df = create_technical_features(train_data, best_params['lookback_window'])
#             train_features = train_df[feature_cols].dropna()
            
#             scaler = StandardScaler()
#             train_features_scaled = scaler.fit_transform(train_features)
            
#             kmeans = KMeans(n_clusters=best_params['n_clusters'], random_state=42, n_init=10)
#             kmeans.fit(train_features_scaled)
            
#             # Apply to test data
#             test_features_scaled = scaler.transform(feature_data)
#             test_clusters = kmeans.predict(test_features_scaled)
            
#             # Generate test signals
#             test_df_clustered = test_df.loc[feature_data.index].copy()
#             test_df_clustered['cluster'] = test_clusters
#             test_df_clustered = generate_cluster_signals(test_df_clustered, best_params['n_clusters'])
            
#             # Calculate test accuracy
#             test_df_clustered['Market_Direction'] = np.where(
#                 test_df_clustered['last_trade_price'].shift(-1) > test_df_clustered['last_trade_price'], 1, 0
#             )
#             test_df_clustered['Correct'] = test_df_clustered['Signal'] == test_df_clustered['Market_Direction']
#             test_accuracy = test_df_clustered['Correct'].mean()
            
#             print(f"Test signal accuracy: {test_accuracy:.2%}")
            
#             # Combine results for overall accuracy
#             combined_results = pd.concat([train_results, test_df_clustered])
#             overall_accuracy = combined_results['Correct'].mean()
#             print(f"Overall signal accuracy: {overall_accuracy:.2%}")
#         else:
#             print("Insufficient test data for clustering")
#             test_accuracy = 0
#             combined_results = train_results
#     else:
#         print("No valid clustering model found")
#         test_accuracy = 0
#         combined_results = pd.DataFrame()
    
#     end_time = time.time()
#     processing_time = round(end_time - start_time, 2)
    
#     # Log best training parameters
#     if best_params['n_clusters'] is not None:
#         log_best_params(
#             file_path='result/clustering_param_log.csv',
#             timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             n_clusters=best_params['n_clusters'],
#             lookback_window=best_params['lookback_window'],
#             features=best_params['features'],
#             accuracy=best_params['accuracy'],
#             processing_time=processing_time,
#             input_start_date=data['time'].min().date(),
#             input_end_date=data['time'].max().date(),
#             signal_method="Clustering"
#         )
    
#     print(f"Processing time: {processing_time} seconds")
    
#     # Optional: Plot cluster analysis
#     if len(combined_results) > 0 and 'cluster' in combined_results.columns:
#         plt.figure(figsize=(15, 10))
        
#         # Plot 1: Price with cluster colors
#         plt.subplot(2, 2, 1)
#         scatter = plt.scatter(range(len(combined_results)), combined_results['last_trade_price'], 
#                             c=combined_results['cluster'], cmap='viridis', alpha=0.6, s=1)
#         plt.title('Price Colored by Cluster')
#         plt.xlabel('Time Index')
#         plt.ylabel('Price')
#         plt.colorbar(scatter)
        
#         # Plot 2: Signal accuracy over time
#         plt.subplot(2, 2, 2)
#         combined_results_sorted = combined_results.sort_values('time') if 'time' in combined_results.columns else combined_results
#         rolling_accuracy = combined_results_sorted['Correct'].rolling(window=500, min_periods=50).mean()
#         plt.plot(rolling_accuracy, color='blue', alpha=0.7)
#         plt.title('Rolling Signal Accuracy (window=500)')
#         plt.ylabel('Accuracy')
#         plt.ylim(0, 1)
#         plt.grid(True, alpha=0.3)
        
#         # Plot 3: Cluster distribution
#         plt.subplot(2, 2, 3)
#         cluster_counts = combined_results['cluster'].value_counts().sort_index()
#         plt.bar(cluster_counts.index, cluster_counts.values)
#         plt.title('Cluster Distribution')
#         plt.xlabel('Cluster')
#         plt.ylabel('Count')
        
#         # Plot 4: Returns by cluster
#         plt.subplot(2, 2, 4)
#         cluster_returns = combined_results.groupby('cluster')['returns'].mean()
#         colors = ['red' if ret < 0 else 'green' for ret in cluster_returns.values]
#         plt.bar(cluster_returns.index, cluster_returns.values, color=colors, alpha=0.7)
#         plt.title('Average Returns by Cluster')
#         plt.xlabel('Cluster')
#         plt.ylabel('Average Return')
#         plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
#         plt.tight_layout()
#         plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score # Not used
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

def create_technical_features(data, lookback_window=20):
    """Create technical indicators for clustering"""
    df = data.copy()
    
    df['returns'] = df['last_trade_price'].pct_change()
    
    # Optimized: Compute rolling object for price once
    rolling_price = df['last_trade_price'].rolling(window=lookback_window)
    
    df['price_ma'] = rolling_price.mean()
    df['price_std'] = rolling_price.std()
    
    df['rsi'] = calculate_rsi(df['last_trade_price'], lookback_window)
    df['momentum'] = df['last_trade_price'] / df['last_trade_price'].shift(lookback_window) - 1
    
    df['volatility'] = df['returns'].rolling(window=lookback_window).std()
    
    # Optimized: Use pre-computed rolling_price
    df['price_range'] = (rolling_price.max() - rolling_price.min()) / df['last_trade_price']
    
    df['price_percentile'] = rolling_price.rank(pct=True)
    df['distance_from_ma'] = (df['last_trade_price'] - df['price_ma']) / df['price_ma']
    
    # Replace inf values that can occur from division by zero (e.g. price_ma is 0)
    # df = df.replace([np.inf, -np.inf], np.nan) # Handled by dropna later, but can be explicit
    
    return df

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    # gain = (delta.where(delta > 0, 0)).rolling(window=window).mean() # Original
    # loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean() # Original
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta).clip(lower=0).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def clustering_strategy_with_tuning(data_input, cluster_range, lookback_range, feature_combinations):
    """
    Clustering-based trading strategy with parameter tuning.
    Optimized loop order and feature calculation.
    """
    best_params = {
        'n_clusters': None, 
        'lookback_window': None, 
        'features': None,
        'accuracy': -np.inf
    }
    best_results_df = pd.DataFrame()
    
    total_combinations = len(cluster_range) * len(lookback_range) * len(feature_combinations)
    current_combination = 0
    
    print(f"Testing {len(lookback_range)} lookback windows × {len(cluster_range)} cluster counts × {len(feature_combinations)} feature sets = {total_combinations} combinations...")
    
    # Optimized loop order:
    for lookback_window in lookback_range:
        # Create technical features ONCE for this lookback_window
        df_with_features = create_technical_features(data_input, lookback_window)
        
        for n_clusters in cluster_range:
            for features_set in feature_combinations:
                current_combination += 1
                if current_combination % 10 == 0 or current_combination == 1 or current_combination == total_combinations:
                    print(f"  Progress: {current_combination}/{total_combinations} ({current_combination/total_combinations*100:.1f}%) | LW: {lookback_window}, NC: {n_clusters}, Feat: {str(features_set[:2])+'...' if len(features_set)>2 else str(features_set)}")

                feature_cols = [col for col in features_set if col in df_with_features.columns]
                
                if len(feature_cols) < 2:
                    continue
                
                feature_data_unscaled = df_with_features[feature_cols].dropna()
                
                if len(feature_data_unscaled) < n_clusters * 10:
                    continue
                
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(feature_data_unscaled)
                
                try:
                    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
                    clusters = kmeans.fit_predict(features_scaled)
                    
                    df_clustered_iter = df_with_features.loc[feature_data_unscaled.index].copy()
                    df_clustered_iter['cluster'] = clusters
                    
                    df_clustered_iter = generate_cluster_signals(df_clustered_iter, n_clusters)
                    
                    df_clustered_iter['Market_Direction'] = np.where(
                        df_clustered_iter['last_trade_price'].shift(-1) > df_clustered_iter['last_trade_price'], 1, 0
                    )
                    
                    temp_accuracy_df = df_clustered_iter[['Signal', 'Market_Direction']].dropna()
                    if temp_accuracy_df.empty:
                        accuracy = 0.0
                    else:
                        accuracy = (temp_accuracy_df['Signal'] == temp_accuracy_df['Market_Direction']).mean()
                    
                    if accuracy > best_params['accuracy']:
                        best_params = {
                            'n_clusters': n_clusters,
                            'lookback_window': lookback_window,
                            'features': features_set,
                            'accuracy': accuracy
                        }
                        best_results_df = df_clustered_iter.copy()
                        
                except Exception as e:
                    # print(f"    Error during clustering: {e}") # Uncomment for debugging
                    continue
    
    return best_params, best_results_df

def generate_cluster_signals(df_input, n_clusters):
    """
    Generate trading signals based on cluster analysis (Vectorized)
    """
    df = df_input.copy()
    
    cluster_returns = df.groupby('cluster')['returns'].mean().fillna(0)
    num_bullish_clusters = max(1, n_clusters // 2)
    bullish_clusters = cluster_returns.nlargest(num_bullish_clusters).index if not cluster_returns.empty else pd.Index([])

    df['prev_cluster'] = df['cluster'].shift(1)
    df['current_is_bullish'] = df['cluster'].isin(bullish_clusters)
    df['prev_is_bullish'] = df['prev_cluster'].isin(bullish_clusters)

    buy_condition = df['current_is_bullish'] & ~df['prev_is_bullish']
    sell_condition = ~df['current_is_bullish'] & df['prev_is_bullish']
    
    df['cluster_momentum_signal'] = np.nan
    df.loc[buy_condition, 'cluster_momentum_signal'] = 1
    df.loc[sell_condition, 'cluster_momentum_signal'] = 0
    df['cluster_momentum_signal'] = df['cluster_momentum_signal'].ffill().fillna(0)
    
    df.drop(columns=['prev_cluster', 'current_is_bullish', 'prev_is_bullish'], inplace=True)
    
    df['cluster_persistence_signal'] = df['cluster'].isin(bullish_clusters).astype(int)
    
    cluster_volatility = df.groupby('cluster')['volatility'].mean().fillna(df['volatility'].mean())
    num_low_vol_clusters = max(1, n_clusters // 2)
    low_vol_clusters = cluster_volatility.nsmallest(num_low_vol_clusters).index if not cluster_volatility.empty else pd.Index([])
        
    df['cluster_vol_signal'] = 0
    condition_low_vol_uptrend = df['cluster'].isin(low_vol_clusters) & (df['returns'].notna() & (df['returns'] > 0))
    df.loc[condition_low_vol_uptrend, 'cluster_vol_signal'] = 1
    
    signal_cols = ['cluster_momentum_signal', 'cluster_persistence_signal', 'cluster_vol_signal']
    df['Signal'] = (df[signal_cols].sum(axis=1) >= 2).astype(int)
    
    return df

def log_best_params(file_path, timestamp, n_clusters, lookback_window, features, accuracy, 
                   processing_time, input_start_date, input_end_date, signal_method):
    """Log the best parameters to CSV"""
    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "n_clusters": n_clusters,
        "lookback_window": lookback_window,
        "features": str(features), # Convert list to string for CSV
        "accuracy": accuracy,
        "processing_time": processing_time,
        "input_date": input_start_date,
        "input_end_date": input_end_date,
        "method": signal_method
    }])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if os.path.exists(file_path):
        new_entry.to_csv(file_path, mode='a', index=False, header=False)
    else:
        new_entry.to_csv(file_path, index=False, header=True)

# Main execution
if __name__ == "__main__":
    data_path = 'data/BTCUSDT-trades-2025-05-20.csv'
    try:
        # For large files, consider using nrows for testing:
        # raw_data = pd.read_csv(data_path, nrows=50000) 
        raw_data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Warning: Data file '{data_path}' not found. Using dummy data for demonstration.")
        dummy_dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=2000, freq='T')) # 'T' for minute
        dummy_prices = 50000 + np.random.randn(2000).cumsum() * 10
        raw_data = pd.DataFrame({'time': dummy_dates.astype(np.int64) // 10**6, 'price': dummy_prices}) # time in ms

    raw_data['time'] = pd.to_datetime(raw_data['time'], unit='ms')
    raw_data = raw_data.rename(columns={'price': 'last_trade_price'})
    raw_data.sort_values('time', inplace=True)
    raw_data.reset_index(drop=True, inplace=True) # Crucial after sorting for consistent indexing
    
    train_size = int(0.8 * len(raw_data))
    train_data = raw_data[:train_size].copy() # Use .copy() to avoid SettingWithCopyWarning
    test_data = raw_data[train_size:].copy()
    
    cluster_range = range(3, 8)
    lookback_range = range(10, 31, 5)
    feature_combinations = [
        ['returns', 'rsi', 'volatility', 'momentum'],
        ['returns', 'price_percentile', 'distance_from_ma', 'volatility'],
        ['rsi', 'momentum', 'price_range', 'distance_from_ma'],
        ['returns', 'rsi', 'momentum', 'price_percentile', 'volatility'],
        ['returns', 'volatility', 'price_range', 'distance_from_ma', 'momentum']
    ]
    
    print("Starting clustering strategy optimization...")
    
    best_params_train, train_results_df = clustering_strategy_with_tuning(
        train_data,
        cluster_range=cluster_range,
        lookback_range=lookback_range,
        feature_combinations=feature_combinations
    )
    
    print("\n" + "="*50)
    print("Best Parameters (from training data):")
    if best_params_train['n_clusters'] is not None:
        print(f"  Clusters: {best_params_train['n_clusters']}")
        print(f"  Lookback Window: {best_params_train['lookback_window']}")
        print(f"  Features: {best_params_train['features']}")
        print(f"  Train signal accuracy: {best_params_train['accuracy']:.2%}")
    else:
        print("  No suitable parameters found during training.")
    print("="*50)
    
    test_accuracy = 0.0
    combined_results = train_results_df.copy()

    if best_params_train['n_clusters'] is not None:
        test_df_with_features = create_technical_features(test_data, best_params_train['lookback_window'])
        best_feature_cols = [col for col in best_params_train['features'] if col in test_df_with_features.columns]

        if not best_feature_cols or len(best_feature_cols) < 2:
            print("Error: Best features not found or insufficient in test_df_with_features.")
        else:
            test_feature_data_unscaled = test_df_with_features[best_feature_cols].dropna()

            if len(test_feature_data_unscaled) >= best_params_train['n_clusters'] * 5 : # Looser check for test data points
                train_df_for_refit = create_technical_features(train_data, best_params_train['lookback_window'])
                train_features_for_refit = train_df_for_refit[best_feature_cols].dropna()

                if train_features_for_refit.empty:
                    print("Error: No training data available for refitting scaler/KMeans with best params.")
                else:
                    scaler = StandardScaler()
                    train_features_scaled_for_refit = scaler.fit_transform(train_features_for_refit)
                    
                    kmeans = KMeans(n_clusters=best_params_train['n_clusters'], init='k-means++', n_init=10, random_state=42)
                    kmeans.fit(train_features_scaled_for_refit)
                    
                    test_features_scaled = scaler.transform(test_feature_data_unscaled)
                    test_clusters = kmeans.predict(test_features_scaled)
                    
                    test_df_clustered = test_df_with_features.loc[test_feature_data_unscaled.index].copy()
                    test_df_clustered['cluster'] = test_clusters
                    test_df_clustered = generate_cluster_signals(test_df_clustered, best_params_train['n_clusters'])
                    
                    test_df_clustered['Market_Direction'] = np.where(
                        test_df_clustered['last_trade_price'].shift(-1) > test_df_clustered['last_trade_price'], 1, 0
                    )
                    
                    temp_accuracy_df_test = test_df_clustered[['Signal', 'Market_Direction']].dropna()
                    test_accuracy = (temp_accuracy_df_test['Signal'] == temp_accuracy_df_test['Market_Direction']).mean() if not temp_accuracy_df_test.empty else 0.0
                    print(f"Test signal accuracy: {test_accuracy:.2%}")

                    # Prepare for combining results
                    if 'Signal' in train_results_df.columns and 'Market_Direction' in train_results_df.columns:
                         train_results_df['Correct'] = (train_results_df['Signal'] == train_results_df['Market_Direction'])
                    
                    test_df_clustered['Correct'] = (test_df_clustered['Signal'] == test_df_clustered['Market_Direction'])
                    
                    cols_to_combine = ['time', 'last_trade_price', 'cluster', 'returns', 'Correct', 'Signal', 'Market_Direction']
                    train_cols = [col for col in cols_to_combine if col in train_results_df.columns]
                    test_cols = [col for col in cols_to_combine if col in test_df_clustered.columns]
                    common_cols_for_concat = list(set(train_cols) & set(test_cols))

                    if common_cols_for_concat:
                        combined_results = pd.concat([
                            train_results_df[common_cols_for_concat], 
                            test_df_clustered[common_cols_for_concat]
                        ], ignore_index=True)
                        
                        if 'Correct' in combined_results.columns:
                            overall_accuracy = combined_results['Correct'].dropna().mean()
                            print(f"Overall signal accuracy (Train+Test): {overall_accuracy:.2%}")
                    else: # Fallback if no common columns (unlikely with this setup)
                        combined_results = train_results_df # Or handle error
            else:
                print("Insufficient test data for clustering with best parameters.")
    else:
        print("No valid clustering model found from training. Skipping test phase.")
    
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    
    if best_params_train['n_clusters'] is not None:
        log_best_params(
            file_path='result/clustering_param_log.csv',
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            n_clusters=best_params_train['n_clusters'],
            lookback_window=best_params_train['lookback_window'],
            features=best_params_train['features'],
            accuracy=best_params_train['accuracy'], # Train accuracy
            processing_time=processing_time,
            input_start_date=raw_data['time'].min().date(),
            input_end_date=raw_data['time'].max().date(),
            signal_method="Clustering_Optimized"
        )
    
    print(f"Total processing time: {processing_time} seconds")
    
    if not combined_results.empty and 'cluster' in combined_results.columns and 'last_trade_price' in combined_results.columns:
        # Sort combined results by time for chronological plotting
        if 'time' in combined_results.columns:
            combined_results = combined_results.sort_values('time').reset_index(drop=True)
        else: # If no time column, use existing index (might be from concat)
            combined_results = combined_results.reset_index(drop=True)

        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(combined_results.index, combined_results['last_trade_price'], 
                            c=combined_results['cluster'], cmap='viridis', alpha=0.6, s=1)
        plt.title('Price Colored by Cluster (Combined Train+Test)')
        plt.xlabel('Time Index (Combined)')
        plt.ylabel('Price')
        if combined_results['cluster'].nunique() > 1: plt.colorbar(scatter)
        
        if 'Correct' in combined_results.columns and not combined_results['Correct'].dropna().empty:
            plt.subplot(2, 2, 2)
            rolling_window_size = min(500, max(10, len(combined_results['Correct'].dropna()) // 10)) # Dynamic window
            rolling_accuracy = combined_results['Correct'].dropna().rolling(window=rolling_window_size, min_periods=max(1, rolling_window_size//10)).mean()
            plt.plot(rolling_accuracy.index, rolling_accuracy, color='blue', alpha=0.7)
            plt.title(f'Rolling Signal Accuracy (window={rolling_window_size})')
            plt.ylabel('Accuracy'); plt.ylim(0, 1); plt.grid(True, alpha=0.3)

        if 'cluster' in combined_results.columns:
            plt.subplot(2, 2, 3)
            cluster_counts = combined_results['cluster'].value_counts().sort_index()
            plt.bar(cluster_counts.index, cluster_counts.values)
            plt.title('Cluster Distribution (Combined)'); plt.xlabel('Cluster'); plt.ylabel('Count')
        
        if 'returns' in combined_results.columns and 'cluster' in combined_results.columns:
            plt.subplot(2, 2, 4)
            cluster_returns_plot = combined_results.groupby('cluster')['returns'].mean().fillna(0)
            if not cluster_returns_plot.empty:
                bar_colors = ['red' if ret < 0 else 'green' for ret in cluster_returns_plot.values]
                plt.bar(cluster_returns_plot.index, cluster_returns_plot.values, color=bar_colors, alpha=0.7)
            plt.title('Average Returns by Cluster (Combined)'); plt.xlabel('Cluster'); plt.ylabel('Average Return')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print("Plotting skipped: No results to plot or essential columns missing.")
