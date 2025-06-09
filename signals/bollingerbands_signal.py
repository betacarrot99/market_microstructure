# signals/bollingerbands_signal.py
import pandas as pd
from signals.base_signal import BaseSignal

class BollingerBands_signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        window = self.params["window"]
        num_std = self.params["num_std"]
        sma = data["price"].rolling(window).mean()
        std = data["price"].rolling(window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        signal = pd.Series(0, index=data.index)
        signal[data["price"] > upper] = -1
        signal[data["price"] < lower] = 1

        return signal

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        best_acc = 0
        best_params = {"window": 50, "num_std": 2.0}
        for window in range(20, 101, 10):
            for num_std in [1.5, 2.0, 2.5]:
                sma = historical_data["price"].rolling(window).mean()
                std = historical_data["price"].rolling(window).std()
                upper = sma + num_std * std
                lower = sma - num_std * std

                signal = pd.Series(0, index=historical_data.index)
                signal[historical_data["price"] > upper] = -1
                signal[historical_data["price"] < lower] = 1

                actual = historical_data["price"].diff().shift(-1).fillna(0).apply(
                    lambda x: 1 if x > 0 else -1 if x < 0 else 0
                )
                acc = (signal == actual).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_params = {"window": window, "num_std": num_std}
        return best_params
