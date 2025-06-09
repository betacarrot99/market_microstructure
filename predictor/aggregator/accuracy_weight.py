# predictor/aggregator/accuracy_weight.py

import numpy as np
from predictor.aggregator.base_aggregator import BaseAggregator

class AccuracyWeightedAggregator(BaseAggregator):
    def __init__(self, accuracy_dict):
        self.accuracy_dict = accuracy_dict

    def aggregate(self, signals: dict) -> float:
        weighted_sum = 0
        total_weight = 0

        for name, signal in signals.items():
            if np.isnan(signal):
                continue
            accuracy = self.accuracy_dict.get(name, 0.5)
            weighted_sum += signal * accuracy
            total_weight += accuracy

        if total_weight == 0:
            return np.nan

        return float(weighted_sum / total_weight)
