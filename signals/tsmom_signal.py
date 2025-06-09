# signals/tsmom_signal.py

import pandas as pd
from signals.base_signal import BaseSignal


class TSMOM_signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        lookback = self.params["lookback"]
        threshold = 0.00001  # fixed threshold value
        threshold_val = data["price"] * threshold

        momentum = data["price"] - data["price"].shift(lookback)

        signal = pd.Series(0, index=data.index)
        signal[momentum > threshold_val] = 1
        signal[momentum < -threshold_val] = -1
        return signal

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        best_acc = 0
        best_lookback = 20
        threshold = 0.00001  # fixed threshold

        for lookback in range(10, 200, 10):
            threshold_val = historical_data["price"] * threshold
            momentum = historical_data["price"] - historical_data["price"].shift(lookback)

            signal = pd.Series(0, index=historical_data.index)
            signal[momentum > threshold_val] = 1
            signal[momentum < -threshold_val] = -1

            price_diff = historical_data["price"].diff().shift(-1).fillna(0)
            threshold_actual = historical_data["price"] * threshold
            actual = pd.Series(0, index=historical_data.index)
            actual[price_diff > threshold_actual] = 1
            actual[price_diff < -threshold_actual] = -1

            acc = (signal == actual).mean()
            if acc > best_acc:
                best_acc = acc
                best_lookback = lookback

        return {"lookback": best_lookback}
