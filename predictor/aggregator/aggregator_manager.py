import pickle
import os
# import xgboost as xgb


from predictor.aggregator.equal_weight import EqualWeightAggregator
from predictor.aggregator.accuracy_weight import AccuracyWeightedAggregator
from predictor.aggregator.majority_vote import MajorityVoteAggregator
# from predictor.aggregator.xgboost_model import XGBoostAggregator

class AggregatorManager:
    def __init__(self, aggregator_names):
        self.aggregator_names = aggregator_names
        self.accuracy_dict = {}  # Placeholder: replace with real-time accuracy if available

        self.aggregators = {}
        for name in aggregator_names:
            if name == "equal_weight":
                self.aggregators[name] = EqualWeightAggregator()
            elif name == "accuracy_weight":
                self.aggregators[name] = AccuracyWeightedAggregator(self.accuracy_dict)
            elif name == "majority_vote":
                self.aggregators[name] = MajorityVoteAggregator()
            # elif name == "xgboost_model":
            #     model_path = "../data/xgboost_model.pkl"
            #     if not os.path.exists(model_path):
            #         raise FileNotFoundError(f"XGBoost model file not found: {model_path}")
            #     with open(model_path, "rb") as f:
            #         bundle = pickle.load(f)
            #         model = bundle["model"]
            #         features = bundle["features"]
            #     self.aggregators[name] = XGBoostAggregator(model, features)

    def aggregate(self, signals: dict) -> dict:
        results = {}
        for name, agg in self.aggregators.items():
            results[name] = agg.aggregate(signals)
        return results
