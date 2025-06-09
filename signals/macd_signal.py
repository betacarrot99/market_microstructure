# signals/macd_signal.py
import pandas as pd
from signals.base_signal import BaseSignal


class MACD_signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        fast = self.params["fast"]
        slow = self.params["slow"]
        signal_span = self.params["signal"]

        ema_fast = data["price"].ewm(span=fast, adjust=False).mean()
        ema_slow = data["price"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()

        cross = (macd_line > signal_line).astype(int)
        signal = cross.diff().fillna(0)
        return signal

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        best_acc = 0
        best_params = {"fast": 12, "slow": 26, "signal": 9}
        for fast in range(10, 21, 2):
            for slow in range(20, 41, 5):
                for signal_span in range(5, 16, 2):
                    ema_fast = historical_data["price"].ewm(span=fast, adjust=False).mean()
                    ema_slow = historical_data["price"].ewm(span=slow, adjust=False).mean()
                    macd_line = ema_fast - ema_slow
                    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
                    cross = (macd_line > signal_line).astype(int)
                    signal = cross.diff().fillna(0)
                    actual = historical_data["price"].diff().shift(-1).fillna(0).apply(
                        lambda x: 1 if x > 0 else -1 if x < 0 else 0
                    )
                    acc = (signal == actual).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {"fast": fast, "slow": slow, "signal": signal_span}
        return best_params
