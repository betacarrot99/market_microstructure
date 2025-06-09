# signals/base_signal.py

from abc import ABC, abstractmethod
import pandas as pd

class BaseSignal(ABC):
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def optimize_params(self, historical_data: pd.DataFrame) -> dict:
        pass
