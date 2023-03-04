from __future__ import annotations
from abc import ABC, abstractmethod

class AIAlgorithmInterface(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def extractFromModel(self, vector):
        pass
