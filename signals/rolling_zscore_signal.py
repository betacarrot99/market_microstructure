# rolling_zscore_signal.py
import pandas as pd
from signals.base_signal import BaseSignal

class RollingZScore_signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:

        window = self.params["window"]
        z_window = self.params["z_window"]

        Z_THRESH = 1.0  # commonly ranges from 1.0 to 2.0 depending on aggressiveness

        mean = data["price"].rolling(window=window).mean()
        std = data["price"].rolling(window=window).std()
        zscore = (data["price"] - mean) / (std + 1e-9)
        rolling_z = zscore.rolling(window=z_window).mean()

        signal = pd.Series(0, index=data.index)
        signal[rolling_z > Z_THRESH] = -1
        signal[rolling_z < -Z_THRESH] = 1
        return signal

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        best_acc = 0
        best_params = {"window": 20, "z_window": 5}  # initial guess
        threshold_label = 0.00001  # used for price change labeling threshold
        Z_THRESH = 1.0  # fixed z-score threshold for signal trigger

        for window in [5, 10, 15, 20, 25]:
            for z_window in [3, 5, 7]:
                mean = historical_data["price"].rolling(window=window).mean()
                std = historical_data["price"].rolling(window=window).std()
                zscore = (historical_data["price"] - mean) / (std + 1e-9)
                rolling_z = zscore.rolling(window=z_window).mean()

                signal = pd.Series(0, index=historical_data.index)
                signal[rolling_z > Z_THRESH] = -1
                signal[rolling_z < -Z_THRESH] = 1

                # Apply threshold to label generation
                diff = historical_data["price"].diff().shift(-1).fillna(0)
                threshold_val = historical_data["price"] * threshold_label
                actual = pd.Series(0, index=historical_data.index)
                actual[diff > threshold_val] = 1
                actual[diff < -threshold_val] = -1

                acc = (signal == actual).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_params = {"window": window, "z_window": z_window}

        return best_params
