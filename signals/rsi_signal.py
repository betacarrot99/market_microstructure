# signals/rsi_signal.py

import pandas as pd
from signals.base_signal import BaseSignal


class RSI_signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        window = self.params["window"]
        delta = data["price"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        signal = rsi.apply(lambda x: 1 if x < 30 else -1 if x > 70 else 0)
        return signal

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        best_acc = 0
        best_window = 14
        for window in range(10, 101, 10):
            delta = historical_data["price"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            signal = rsi.apply(lambda x: 1 if x < 30 else -1 if x > 70 else 0)

            threshold = 0.00001  # or whatever your fixed project-wide threshold is
            diff = historical_data["price"].diff().shift(-1).fillna(0)
            threshold_val = historical_data["price"] * threshold

            actual = pd.Series(0, index=historical_data.index)
            actual[diff > threshold_val] = 1
            actual[diff < -threshold_val] = -1

            acc = (signal == actual).mean()
            if acc > best_acc:
                best_acc = acc
                best_window = window
        return {"window": best_window}
