# twap_signal.py
import pandas as pd
from signals.base_signal import BaseSignal

class TWAP_signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        twap = data["price"].expanding().mean()
        deviation_threshold = self.params.get("deviation", 0.00001)  # 0.0001%
        threshold_value = twap * deviation_threshold

        signal = pd.Series(0, index=data.index)
        signal[data["price"] > twap + threshold_value] = 1
        signal[data["price"] < twap - threshold_value] = -1
        return signal

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
            return {}  # no params for TWAP
