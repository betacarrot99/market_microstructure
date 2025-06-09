from abc import ABC, abstractmethod

class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(self, signals: dict) -> float:
        pass
