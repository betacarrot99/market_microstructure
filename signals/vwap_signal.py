
# vwap_signal.py
import pandas as pd
from signals.base_signal import BaseSignal

class VWAP_signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        vwap = (data["price"] * data["volume"]).cumsum() / data["volume"].cumsum()
        deviation_threshold = self.params.get("deviation", 0.00001)  # 0.0001%
        threshold_value = vwap * deviation_threshold

        signal = pd.Series(0, index=data.index)
        signal[data["price"] > vwap + threshold_value] = 1
        signal[data["price"] < vwap - threshold_value] = -1
        return signal

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        return {}  # no params for VWAP
