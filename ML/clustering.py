import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
import json

# Import from the new module
from trading_indicators import TradingIndicators 

warnings.filterwarnings('ignore')

start_time = time.time()

# Functions create_technical_features, calculate_rsi, generate_cluster_signals
# are now part of TradingIndicators class and will be called using TradingIndicators.method_name()

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
    
    for lookback_window in lookback_range:
        # Use TradingIndicators for feature creation
        df_with_features = TradingIndicators.create_technical_features(data_input, lookback_window)
        
        for n_clusters in cluster_range:
            for features_set in feature_combinations:
                current_combination += 1
                if current_combination % 10 == 0 or current_combination == 1 or current_combination == total_combinations:
                    print(f"  Progress: {current_combination}/{total_combinations} ({current_combination/total_combinations*100:.1f}%) | LW: {lookback_window}, NC: {n_clusters}, Feat: {str(features_set[:2])+'...' if len(features_set)>2 else str(features_set)}")

                feature_cols = [col for col in features_set if col in df_with_features.columns]
                
                if len(feature_cols) < 2:
                    continue
                
                feature_data_unscaled = df_with_features[feature_cols].dropna()
                
                # Ensure enough data points for clustering after NaNs from features are dropped
                if len(feature_data_unscaled) < n_clusters * 10: # Min 10 points per cluster
                    continue
                
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(feature_data_unscaled)
                
                try:
                    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
                    clusters = kmeans.fit_predict(features_scaled)
                    
                    df_clustered_iter = df_with_features.loc[feature_data_unscaled.index].copy()
                    df_clustered_iter['cluster'] = clusters
                    
                    # Use TradingIndicators for signal generation
                    df_clustered_iter = TradingIndicators.generate_cluster_signals(df_clustered_iter, n_clusters)
                    
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
                    # print(f"    Error during clustering for LW:{lookback_window}, NC:{n_clusters}, Feat:{features_set} - {e}")
                    continue
    
    return best_params, best_results_df

def log_best_params(file_path, timestamp, n_clusters, lookback_window, features, accuracy, 
                   processing_time, input_start_date, input_end_date, signal_method):
    """Log the best parameters to CSV"""
    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "n_clusters": n_clusters,
        "lookback_window": lookback_window,
        "features": str(features), 
        "accuracy": accuracy,
        "processing_time": processing_time,
        "input_date": input_start_date,
        "input_end_date": input_end_date,
        "method": signal_method
    }])
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if os.path.exists(file_path):
        new_entry.to_csv(file_path, mode='a', index=False, header=False)
    else:
        new_entry.to_csv(file_path, index=False, header=True)

# Main execution
if __name__ == "__main__":

    # data_path = 'data/BTCUSDT-trades-latest-2days.csv' # New path for the downloaded data
    # data_path = 'data/BTCUSDT-trades-previous-day.csv' # For testing with previous day data
    # Or, if we decide to use resampled data:
    # data_path = 'data/BTCUSDT-trades-latest-2days-resampled-1s.csv' 
    # data_path = 'data/ETHUSDT-trades-latest-2days-resampled-1s.csv' 
    data_path = 'data/BTCUSDT-trades-latest-2days-resampled-100ms.csv'
    try:
        raw_data = pd.read_csv(data_path)
        # raw_data = pd.read_csv(data_path, nrows=50000) # For faster testing
    except FileNotFoundError:
        print(f"Warning: Data file '{data_path}' not found. Using dummy data for demonstration.")
        dummy_dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10000, freq='S')) # Use seconds for more data points
        dummy_prices = 50000 + np.random.randn(10000).cumsum() * 2
        raw_data = pd.DataFrame({'time': dummy_dates.astype(np.int64) // 10**6, 'price': dummy_prices})

    raw_data['time'] = pd.to_datetime(raw_data['time'], unit='ms')
    raw_data = raw_data.rename(columns={'price': 'last_trade_price'})
    raw_data.sort_values('time', inplace=True)
    raw_data.reset_index(drop=True, inplace=True) 
    
    # Reduce data size for faster tuning if using full dataset is too slow initially
    # raw_data = raw_data.sample(n=min(len(raw_data), 100000), random_state=42).sort_values('time').reset_index(drop=True)


    train_size = int(0.8 * len(raw_data))
    train_data = raw_data[:train_size].copy() 
    test_data = raw_data[train_size:].copy()
    
    if len(train_data) < 100: # Basic check for minimal data
        print("Error: Not enough training data to proceed.")
        exit()

    cluster_range = range(3, 6)         # Reduced for faster example run: 3 to 5 clusters
    lookback_range = range(15, 31, 10)  # Reduced for faster example run: 15, 25
    
    feature_combinations = [
        ['returns', 'rsi', 'volatility', 'momentum'],
        ['returns', 'price_percentile', 'distance_from_ma', 'volatility'],
        ['rsi', 'momentum', 'price_range', 'distance_from_ma'],
        ['returns', 'rsi', 'momentum', 'price_percentile', 'volatility'],
        ['returns', 'volatility', 'price_range', 'distance_from_ma', 'momentum']
    ] # Reduced feature sets for faster example
    
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
    
    # Model saving logic
    MODEL_ARTIFACTS_DIR = "model_artifacts"
    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
    KMEANS_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "best_kmeans_model.joblib")
    SCALER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "best_scaler.joblib")
    CLUSTERING_CONFIG_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "best_clustering_config.json")
    
    final_kmeans_model = None
    final_scaler = None

    if best_params_train['n_clusters'] is not None:
        print("\nTraining final model on full training data with best parameters...")
        full_train_df_for_final_model = TradingIndicators.create_technical_features(train_data.copy(), best_params_train['lookback_window'])
        final_model_feature_cols = [col for col in best_params_train['features'] if col in full_train_df_for_final_model.columns]
        
        final_model_train_features_unscaled = full_train_df_for_final_model[final_model_feature_cols].dropna()

        if not final_model_train_features_unscaled.empty and len(final_model_train_features_unscaled) >= best_params_train['n_clusters'] * 5:
            final_scaler = StandardScaler()
            final_train_features_scaled = final_scaler.fit_transform(final_model_train_features_unscaled)
            
            final_kmeans_model = KMeans(n_clusters=best_params_train['n_clusters'], init='k-means++', n_init=10, random_state=42)
            final_kmeans_model.fit(final_train_features_scaled)

            joblib.dump(final_scaler, SCALER_PATH)
            joblib.dump(final_kmeans_model, KMEANS_MODEL_PATH)
            
            clustering_config_to_save = {
                'lookback_window': best_params_train['lookback_window'],
                'features': best_params_train['features'],
                'n_clusters': int(best_params_train['n_clusters']) # Ensure native int
            }
            with open(CLUSTERING_CONFIG_PATH, 'w') as f:
                json.dump(clustering_config_to_save, f, indent=4)
            print(f"Saved final scaler to {SCALER_PATH}")
            print(f"Saved final KMeans model to {KMEANS_MODEL_PATH}")
            print(f"Saved final clustering config to {CLUSTERING_CONFIG_PATH}")
        else:
            print("Could not train and save final model: insufficient data or other issue.")
    
    # Test evaluation
    test_accuracy = 0.0
    combined_results = train_results_df.copy() if not train_results_df.empty else pd.DataFrame()

    if final_kmeans_model and final_scaler and best_params_train['n_clusters'] is not None and not test_data.empty:
        print("\nEvaluating final model on test data...")
        test_df_with_features = TradingIndicators.create_technical_features(test_data.copy(), best_params_train['lookback_window'])
        best_feature_cols = [col for col in best_params_train['features'] if col in test_df_with_features.columns]

        if best_feature_cols and len(best_feature_cols) >= 2:
            test_feature_data_unscaled = test_df_with_features[best_feature_cols].dropna()
            if len(test_feature_data_unscaled) >= best_params_train['n_clusters'] * 2: # Relaxed condition for test
                test_features_scaled = final_scaler.transform(test_feature_data_unscaled)
                test_clusters = final_kmeans_model.predict(test_features_scaled)
                
                test_df_clustered = test_df_with_features.loc[test_feature_data_unscaled.index].copy()
                test_df_clustered['cluster'] = test_clusters
                test_df_clustered = TradingIndicators.generate_cluster_signals(test_df_clustered, best_params_train['n_clusters'])
                
                test_df_clustered['Market_Direction'] = np.where(
                    test_df_clustered['last_trade_price'].shift(-1) > test_df_clustered['last_trade_price'], 1, 0
                )
                
                temp_accuracy_df_test = test_df_clustered[['Signal', 'Market_Direction']].dropna()
                test_accuracy = (temp_accuracy_df_test['Signal'] == temp_accuracy_df_test['Market_Direction']).mean() if not temp_accuracy_df_test.empty else 0.0
                print(f"Test signal accuracy (using final model): {test_accuracy:.2%}")
                
                # Combine results logic
                if 'Signal' in train_results_df.columns and 'Market_Direction' in train_results_df.columns:
                     if 'Correct' not in train_results_df.columns:
                        train_results_df['Correct'] = (train_results_df['Signal'] == train_results_df['Market_Direction'])
                
                if 'Signal' in test_df_clustered.columns and 'Market_Direction' in test_df_clustered.columns:
                    test_df_clustered['Correct'] = (test_df_clustered['Signal'] == test_df_clustered['Market_Direction'])
                
                cols_to_combine = ['time', 'last_trade_price', 'cluster', 'returns', 'Correct', 'Signal', 'Market_Direction']
                train_valid_cols = [col for col in cols_to_combine if col in train_results_df.columns]
                test_valid_cols = [col for col in cols_to_combine if col in test_df_clustered.columns]
                
                common_cols = list(set(train_valid_cols) & set(test_valid_cols))

                if common_cols:
                    combined_results = pd.concat([
                        train_results_df[common_cols] if not train_results_df.empty else pd.DataFrame(columns=common_cols),
                        test_df_clustered[common_cols] if not test_df_clustered.empty else pd.DataFrame(columns=common_cols)
                    ], ignore_index=True)
                    
                    if 'Correct' in combined_results.columns and not combined_results['Correct'].dropna().empty:
                        overall_accuracy = combined_results['Correct'].dropna().mean()
                        print(f"Overall signal accuracy (Train+Test): {overall_accuracy:.2%}")
            else:
                print("Insufficient test data after feature calculation for clustering evaluation.")
        else:
            print("Best features not found or insufficient in test_df_with_features for test evaluation.")
    elif best_params_train['n_clusters'] is None:
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
            accuracy=best_params_train['accuracy'],
            processing_time=processing_time,
            input_start_date=raw_data['time'].min().date() if not raw_data.empty else 'N/A',
            input_end_date=raw_data['time'].max().date() if not raw_data.empty else 'N/A',
            signal_method="Clustering_Optimized_V2"
        )
    
    print(f"Total processing time: {processing_time} seconds")
    
    # Plotting (ensure combined_results is not empty and has necessary columns)
    if not combined_results.empty and 'cluster' in combined_results.columns and 'last_trade_price' in combined_results.columns:
        if 'time' in combined_results.columns:
            combined_results = combined_results.sort_values('time').reset_index(drop=True)
        else:
            combined_results = combined_results.reset_index(drop=True)

        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.scatter(combined_results.index, combined_results['last_trade_price'], 
                    c=combined_results['cluster'], cmap='viridis', alpha=0.6, s=1)
        plt.title('Price Colored by Cluster (Combined Train+Test)')
        plt.xlabel('Time Index (Combined)'); plt.ylabel('Price')
        if combined_results['cluster'].nunique() > 1: plt.colorbar(label='Cluster')
        
        if 'Correct' in combined_results.columns and not combined_results['Correct'].dropna().empty:
            plt.subplot(2, 2, 2)
            rolling_window_size = min(500, max(50, len(combined_results['Correct'].dropna()) // 10))
            rolling_accuracy = combined_results['Correct'].dropna().rolling(window=rolling_window_size, min_periods=max(1, rolling_window_size//10)).mean()
            plt.plot(rolling_accuracy.index, rolling_accuracy, color='blue', alpha=0.7)
            plt.title(f'Rolling Signal Accuracy (window={rolling_window_size})')
            plt.ylabel('Accuracy'); plt.ylim(0, 1); plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        cluster_counts = combined_results['cluster'].value_counts().sort_index()
        plt.bar(cluster_counts.index, cluster_counts.values)
        plt.title('Cluster Distribution (Combined)'); plt.xlabel('Cluster'); plt.ylabel('Count')
        
        if 'returns' in combined_results.columns :
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
        print("Plotting skipped: No results or essential columns missing.")