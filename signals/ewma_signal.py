# signals/ewma_signal.py

import pandas as pd
from signals.base_signal import BaseSignal

class EWMA_signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        span = self.params["span"]
        ewma = data["price"].ewm(span=span, adjust=False).mean()
        above_ewma = (data["price"] > ewma).astype(int)
        signal = above_ewma.diff().fillna(0)
        return signal

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        best_acc = 0
        best_span = 20
        for span in range(10, 101, 10):
            ewma = historical_data["price"].ewm(span=span, adjust=False).mean()
            above_ewma = (historical_data["price"] > ewma).astype(int)
            signal = above_ewma.diff().fillna(0)
            actual = historical_data["price"].diff().shift(-1).fillna(0).apply(
                lambda x: 1 if x > 0 else -1 if x < 0 else 0
            )
            acc = (signal == actual).mean()
            if acc > best_acc:
                best_acc = acc
                best_span = span
        return {"span": best_span}
