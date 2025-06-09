# predictor/aggregator/equal_weight.py

import pandas as pd
import numpy as np
from predictor.aggregator.base_aggregator import BaseAggregator

class EqualWeightAggregator(BaseAggregator):
    def aggregate(self, signals: dict) -> float:
        valid_signals = [s for s in signals.values() if not np.isnan(s)]
        if not valid_signals:
            return np.nan
        return float(np.mean(valid_signals))
