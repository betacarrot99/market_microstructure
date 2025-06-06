import pandas as pd
import numpy as np

class TradingIndicators:

    @staticmethod
    def calculate_rsi(prices, window=14):
        """Calculate Relative Strength Index"""
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = (-delta).clip(lower=0).rolling(window=window).mean()
        
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan) # Handle division by zero if loss is 0
        
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def create_technical_features(data, lookback_window=20):
        """Create technical indicators for clustering.
           Assumes 'data' DataFrame has a 'last_trade_price' column.
        """
        df = data.copy()
    
        if 'last_trade_price' not in df.columns:
            # This case should ideally be handled before calling this function
            raise ValueError("Input DataFrame to create_technical_features must contain 'last_trade_price' column.")

        df['returns'] = df['last_trade_price'].pct_change()
        
        rolling_price = df['last_trade_price'].rolling(window=lookback_window)
        
        df['price_ma'] = rolling_price.mean()
        df['price_std'] = rolling_price.std()
        
        df['rsi'] = TradingIndicators.calculate_rsi(df['last_trade_price'], lookback_window)
        
        # Ensure shift has enough data; produces NaN for first 'lookback_window' entries
        df['momentum'] = df['last_trade_price'] / df['last_trade_price'].shift(lookback_window) - 1
        
        df['volatility'] = df['returns'].rolling(window=lookback_window).std()
        
        price_max = rolling_price.max()
        price_min = rolling_price.min()
        # Avoid division by zero if last_trade_price is 0
        df['price_range'] = (price_max - price_min).divide(df['last_trade_price'].replace(0, np.nan))
        
        df['price_percentile'] = df['last_trade_price'].rolling(window=lookback_window).rank(pct=True)
        
        # Avoid division by zero if price_ma is 0
        df['distance_from_ma'] = (df['last_trade_price'] - df['price_ma']).divide(df['price_ma'].replace(0, np.nan))
        
        # Replace any remaining inf values (e.g., from division by a very small number if not 0)
        df = df.replace([np.inf, -np.inf], np.nan)
        return df
   
    @staticmethod
    def generate_cluster_signals(df_input, n_clusters):
        """
        Generate trading signals based on cluster analysis (Vectorized)
        """
        df = df_input.copy()
        if 'cluster' not in df.columns or df['cluster'].isnull().all():
            df['Signal'] = 0
            df['cluster_momentum_signal'] = 0
            df['cluster_persistence_signal'] = 0
            df['cluster_vol_signal'] = 0
            return df
        
        # Ensure required columns exist for groupby, fillna for robustness
        if 'returns' not in df.columns: df['returns'] = 0.0
        if 'volatility' not in df.columns: df['volatility'] = 0.0

        cluster_returns = df.groupby('cluster')['returns'].mean().fillna(0)
        num_bullish_clusters = max(1, int(n_clusters // 2))
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
        
        df.drop(columns=['prev_cluster', 'current_is_bullish', 'prev_is_bullish'], inplace=True, errors='ignore')
        
        df['cluster_persistence_signal'] = df['cluster'].isin(bullish_clusters).astype(int)
        
        global_volatility_mean = df['volatility'].mean()
        if pd.isna(global_volatility_mean): global_volatility_mean = 0.0 # Fallback

        cluster_volatility = df.groupby('cluster')['volatility'].mean().fillna(global_volatility_mean)
        num_low_vol_clusters = max(1, int(n_clusters // 2))
        low_vol_clusters = cluster_volatility.nsmallest(num_low_vol_clusters).index if not cluster_volatility.empty else pd.Index([])
            
        df['cluster_vol_signal'] = 0
        if 'returns' in df.columns:
            condition_low_vol_uptrend = df['cluster'].isin(low_vol_clusters) & (df['returns'].notna() & (df['returns'] > 0))
            df.loc[condition_low_vol_uptrend, 'cluster_vol_signal'] = 1
        
        signal_cols = ['cluster_momentum_signal', 'cluster_persistence_signal', 'cluster_vol_signal']
        
        existing_signal_cols = [col for col in signal_cols if col in df.columns]
        if existing_signal_cols:
            df['Signal'] = (df[existing_signal_cols].sum(axis=1) >= 2).astype(int)
        else:
            df['Signal'] = 0 
        return df
