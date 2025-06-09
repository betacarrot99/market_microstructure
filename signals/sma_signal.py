# signals/sma_signal.py

import pandas as pd
from signals.base_signal import BaseSignal

class SMA_signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        window = self.params["window"]
        sma = data["price"].rolling(window).mean()
        above_sma = (data["price"] > sma).astype(int)
        signal = above_sma.diff().fillna(0)
        return signal

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        best_acc = 0
        best_window = 20
        for window in range(10, 101, 10):
            sma = historical_data["price"].rolling(window).mean()
            above_sma = (historical_data["price"] > sma).astype(int)
            signal = above_sma.diff().fillna(0)
            actual = historical_data["price"].diff().shift(-1).fillna(0).apply(
                lambda x: 1 if x > 0 else -1 if x < 0 else 0
            )
            acc = (signal == actual).mean()
            if acc > best_acc:
                best_acc = acc
                best_window = window
        return {"window": best_window}
