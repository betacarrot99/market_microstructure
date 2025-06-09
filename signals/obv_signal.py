# obv_signal.py
import pandas as pd
from signals.base_signal import BaseSignal

class OBV_signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        threshold = 0.00001  # example value
        diff = data["price"].diff().fillna(0)
        threshold_val = data["price"] * threshold

        direction = pd.Series(0, index=data.index)
        direction[diff > threshold_val] = 1
        direction[diff < -threshold_val] = -1

        obv = (data["volume"] * direction).cumsum()

        obv_smooth = obv.rolling(window=self.params["window"]).mean()
        signal = pd.Series(0, index=data.index)
        signal[obv > obv_smooth] = 1
        signal[obv < obv_smooth] = -1
        return signal

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        best_acc = 0
        best_window = 20
        threshold = 0.00001

        diff = historical_data["price"].diff().fillna(0)
        threshold_val = historical_data["price"] * threshold

        direction = pd.Series(0, index=historical_data.index)
        direction[diff > threshold_val] = 1
        direction[diff < -threshold_val] = -1

        for window in range(10, 101, 10):
            obv = (historical_data["volume"] * direction).cumsum()
            obv_smooth = obv.rolling(window=window).mean()

            signal = pd.Series(0, index=historical_data.index)
            signal[obv > obv_smooth] = 1
            signal[obv < obv_smooth] = -1

            actual_diff = historical_data["price"].diff().shift(-1).fillna(0)
            actual_threshold_val = historical_data["price"] * threshold
            actual = pd.Series(0, index=historical_data.index)
            actual[actual_diff > actual_threshold_val] = 1
            actual[actual_diff < -actual_threshold_val] = -1

            acc = (signal == actual).mean()
            if acc > best_acc:
                best_acc = acc
                best_window = window

        return {"window": best_window}
