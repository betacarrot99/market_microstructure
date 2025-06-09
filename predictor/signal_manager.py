# signal_manager.py

import inspect
import pandas as pd
import importlib
import pkgutil
from signals.base_signal import BaseSignal
import signals  # assuming signals is a package with __init__.py

class SignalManager:
    def __init__(self):
        self.signal_classes = self._discover_signals()

    def _discover_signals(self):
        signal_classes = {}
        for loader, module_name, _ in pkgutil.walk_packages(signals.__path__):
            module = importlib.import_module(f"signals.{module_name}")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if name.endswith("_signal") and issubclass(obj, BaseSignal) and obj is not BaseSignal:
                    signal_classes[name] = obj
        return signal_classes

    def optimize_all(self, historical_data: pd.DataFrame):
        optimized = {}
        for name, SignalClass in self.signal_classes.items():
            instance = SignalClass(params={})
            optimized[name] = instance.optimize_params(historical_data)
        return optimized

    def generate_all_signals(self, data: pd.DataFrame, optimized_params: dict):
        signals_output = {}
        for name, SignalClass in self.signal_classes.items():
            params = optimized_params.get(name, {})
            instance = SignalClass(params=params)
            signals_output[name] = instance.generate_signal(data)
        return pd.DataFrame(signals_output)

# Example usage:
# manager = SignalManager()
# hist_data = pd.read_csv("btc_trade_data.csv")  # must contain 'price'
# opt_params = manager.optimize_all(hist_data)
# signal_df = manager.generate_all_signals(hist_data, opt_params)
# print(signal_df.tail())
