import pandas as pd
import numpy as np

class SignalGenerator:
    """
    Encapsulates all signal generation methods for various strategies.
    """
    @staticmethod
    def sma(df: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
        ind_short = df['last_trade_price'].rolling(window=short_window).mean()
        ind_long  = df['last_trade_price'].rolling(window=long_window).mean()
        return np.where(ind_short > ind_long, 1, 0)

    @staticmethod
    def ewma(df: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
        ind_short = df['last_trade_price'].ewm(span=short_window, adjust=False).mean()
        ind_long  = df['last_trade_price'].ewm(span=long_window, adjust=False).mean()
        return np.where(ind_short > ind_long, 1, 0)

    @staticmethod
    def tsmom(df: pd.DataFrame, lookback: int) -> pd.Series:
        return np.where(df['last_trade_price'] > df['last_trade_price'].shift(lookback), 1, 0)

    @staticmethod
    def rsi(df: pd.DataFrame, lookback: int) -> pd.Series:
        delta = df['last_trade_price'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=lookback).mean()
        avg_loss = loss.rolling(window=lookback).mean()
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        sig = np.where(rsi < 30, 1, np.where(rsi > 70, 0, np.nan))
        return pd.Series(sig).ffill().fillna(0)

    @staticmethod
    def rsi_momentum(df: pd.DataFrame, lookback: int) -> pd.Series:
        delta = df['last_trade_price'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=lookback).mean()
        avg_loss = loss.rolling(window=lookback).mean()
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        sig = np.where(rsi > 55, 1, np.where(rsi < 45, 0, np.nan))
        return pd.Series(sig).ffill().fillna(0)

    @staticmethod
    def bb(df: pd.DataFrame, lookback: int, std_dev: float) -> pd.Series:
        ma = df['last_trade_price'].rolling(window=lookback).mean()
        std = df['last_trade_price'].rolling(window=lookback).std()
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        sig = np.where(df['last_trade_price'] < lower, 1,
                       np.where(df['last_trade_price'] > upper, 0, np.nan))
        return pd.Series(sig).ffill().fillna(0)

    @staticmethod
    def obv(df: pd.DataFrame, lookback: int) -> pd.Series:
        vol = df['qty'] if 'qty' in df.columns else 1
        obv = (np.sign(df['last_trade_price'].diff()) * vol).cumsum()
        obv_ma = obv.rolling(window=lookback).mean()
        return np.where(obv > obv_ma, 1, 0)

    @staticmethod
    def macd(df: pd.DataFrame, short_window: int, long_window: int, lookback: int) -> pd.Series:
        ema_fast = df['last_trade_price'].ewm(span=short_window, adjust=False).mean()
        ema_slow = df['last_trade_price'].ewm(span=long_window, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=lookback, adjust=False).mean()
        return np.where(macd_line > signal_line, 1, 0)

    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int, d_period: int) -> pd.Series:
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k = 100 * (df['last_trade_price'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        sig = np.where((k < 20) & (k > d), 1,
                       np.where((k > 80) & (k < d), 0, np.nan))
        return pd.Series(sig).ffill().fillna(0)

    @staticmethod
    def atr(df: pd.DataFrame, lookback: int, std_dev: float) -> pd.Series:
        high, low, close = df['high'], df['low'], df['last_trade_price']
        prev_close = close.shift(1)
        tr = pd.concat([high - low,
                        (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=lookback).mean()
        upper = prev_close + std_dev * atr
        lower = prev_close - std_dev * atr
        sig = np.where(close > upper, 1,
                       np.where(close < lower, 0, np.nan))
        return pd.Series(sig).ffill().fillna(0)

    @staticmethod
    def donchian(df: pd.DataFrame, lookback: int) -> pd.Series:
        upper = df['high'].rolling(window=lookback).max()
        lower = df['low'].rolling(window=lookback).min()
        sig = np.where(df['last_trade_price'] > upper.shift(1), 1,
                       np.where(df['last_trade_price'] < lower.shift(1), 0, np.nan))
        return pd.Series(sig).ffill().fillna(0)

    @staticmethod
    def zscore(df: pd.DataFrame, lookback: int, std_dev: float) -> pd.Series:
        mean = df['last_trade_price'].rolling(window=lookback).mean()
        sd = df['last_trade_price'].rolling(window=lookback).std()
        z = (df['last_trade_price'] - mean) / sd
        sig = np.where(z < -std_dev, 1,
                       np.where(z > std_dev, 0, np.nan))
        return pd.Series(sig).ffill().fillna(0)

    @classmethod
    def compute_signal(cls, df: pd.DataFrame, strategy: str,
                       short_window: int=None, long_window: int=None,
                       lookback: int=None, std_dev: float=None) -> pd.Series:
        # dispatch based on strategy name
        strat = strategy.lower()
        if strat == 'sma':
            return cls.sma(df, short_window, long_window)
        if strat == 'ewma':
            return cls.ewma(df, short_window, long_window)
        if strat == 'tsmom':
            return cls.tsmom(df, lookback)
        if strat == 'rsi':
            return cls.rsi(df, lookback)
        if strat == 'rsi_momentum':
            return cls.rsi_momentum(df, lookback)
        if strat == 'bb':
            return cls.bb(df, lookback, std_dev)
        if strat == 'obv':
            return cls.obv(df, lookback)
        if strat == 'macd':
            return cls.macd(df, short_window, long_window, lookback)
        if strat == 'stochastic':
            return cls.stochastic(df, short_window, long_window)
        if strat == 'atr':
            return cls.atr(df, lookback, std_dev)
        if strat == 'donchian':
            return cls.donchian(df, lookback)
        if strat == 'zscore':
            return cls.zscore(df, lookback, std_dev)
        raise ValueError(f"Unknown strategy: {strategy}")

class PnlCalculator:
    """
    Computes PnL series from signals for backtesting or live usage.
    """
    def __init__(self, data: pd.DataFrame, price_col: str = 'last_trade_price'):
        # allow loading DataFrames with either 'price' or 'last_trade_price'
        self.df = data.copy()
        if price_col in self.df.columns and price_col != 'last_trade_price':
            self.df.rename(columns={price_col: 'last_trade_price'}, inplace=True)

    def compute_pnl_series(self, strategy: str,
                           short_window: int=None, long_window: int=None,
                           lookback: int=None, std_dev: float=None) -> np.ndarray:
        """
        Returns an array of PnL values for each time step, analogous to weight_accuracy_live outputs.
        """
        df = self.df.copy()
        # generate discrete signals
        df['Signal'] = SignalGenerator.compute_signal(
            df, strategy, short_window, long_window, lookback, std_dev
        )
        # positions shifted one step for execution
        df['Position'] = df['Signal'].diff().shift(1).fillna(0)
        # price and returns
        df['Next_Price'] = df['last_trade_price'].shift(-1)
        df['log_return'] = np.log(df['Next_Price'] / df['last_trade_price'])
        # PnL per step
        pnl = df['Position'] * df['log_return']
        # return as plain numpy array for easy integration
        return pnl.values
