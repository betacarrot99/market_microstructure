# signals/mean_reversion_signal.py
import pandas as pd
from signals.base_signal import BaseSignal


class MeanReversion_signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        window = self.params["window"]
        threshold = self.params["threshold"]
        mean = data["price"].rolling(window).mean()
        std = data["price"].rolling(window).std()
        zscore = (data["price"] - mean) / (std + 1e-9)
        signal = zscore.apply(lambda x: -1 if x > threshold else 1 if x < -threshold else 0)
        return signal

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        best_acc = 0
        best_params = {"window": 20, "threshold": 1.0}
        for window in range(10, 101, 10):
            for threshold in [0.5, 1.0, 1.5, 2.0]:
                mean = historical_data["price"].rolling(window).mean()
                std = historical_data["price"].rolling(window).std()
                zscore = (historical_data["price"] - mean) / (std + 1e-9)
                signal = zscore.apply(lambda x: -1 if x > threshold else 1 if x < -threshold else 0)
                actual = historical_data["price"].diff().shift(-1).fillna(0).apply(
                    lambda x: 1 if x > 0 else -1 if x < 0 else 0
                )
                acc = (signal == actual).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_params = {"window": window, "threshold": threshold}
        return best_params
