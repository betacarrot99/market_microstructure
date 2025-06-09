# backtester.py

import pandas as pd

class SignalBacktester:
    def __init__(self, signal_df, price_series, threshold=0.0):
        self.signal_df = signal_df.reset_index(drop=True)
        self.true_labels = self._generate_true_labels(price_series, threshold)

    def _generate_true_labels(self, price, threshold):
        diff = price.diff().shift(-1).fillna(0)
        threshold_val = price * threshold

        labels = pd.Series(0, index=price.index)
        labels[diff > threshold_val] = 1
        labels[diff < -threshold_val] = -1

        return labels.reset_index(drop=True)

    def accuracy(self):
        return {
            name: (self.true_labels == preds).mean()
            for name, preds in self.signal_df.items()
        }

    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        return {
            name: confusion_matrix(self.true_labels, preds, labels=[-1, 0, 1])
            for name, preds in self.signal_df.items()
        }

    def signal_summary(self):
        return self.signal_df.apply(lambda s: s.value_counts().reindex([-1, 0, 1], fill_value=0))
