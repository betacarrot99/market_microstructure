# predictor/aggregator/majority_vote.py
import numpy as np
from collections import Counter
from predictor.aggregator.base_aggregator import BaseAggregator

class MajorityVoteAggregator(BaseAggregator):
    def aggregate(self, signals: dict) -> float:
        valid_signals = [int(s) for s in signals.values() if not np.isnan(s)]
        if not valid_signals:
            return np.nan

        count = Counter(valid_signals)
        majority = count.most_common(1)[0][0]
        return float(majority)
