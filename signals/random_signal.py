# signals/random_signal.py

import pandas as pd
import numpy as np
from signals.base_signal import BaseSignal

class Random_Signal(BaseSignal):
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        np.random.seed(self.params.get("seed", 42))
        return pd.Series(np.random.choice([-1, 0, 1], size=len(data)), index=data.index)

    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        return {"seed": 42}