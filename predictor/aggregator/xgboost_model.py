# xgboost_model.py
import pandas as pd
import xgboost_model as xgb
from predictor.aggregator.base_aggregator import BaseAggregator

import numpy as np
import xgboost as xgb
from predictor.aggregator.base_aggregator import BaseAggregator

class XGBoostAggregator(BaseAggregator):
    def __init__(self, model: xgb.XGBClassifier, feature_order: list):
        self.model = model
        self.feature_order = feature_order

    def aggregate(self, signals: dict) -> float:
        features = []
        for name in self.feature_order:
            val = signals.get(name, np.nan)
            features.append(np.nan if np.isnan(val) else val)

        if any(np.isnan(features)):
            return np.nan

        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)
        return float(prediction[0])
