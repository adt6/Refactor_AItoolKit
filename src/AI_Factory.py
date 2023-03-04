from __future__ import annotations
from abc import ABC, abstractmethod
from src.AI_Algorithm import AIAlgorithmInterface


class AIFactory(ABC):
    @abstractmethod
    def createAIMethod(self, trainingDataSet, allDataSet):
        pass

    def getAIMethod(self, trainingDataSet, allDataSet) -> AIAlgorithmInterface:
        method = self.createAIMethod(trainingDataSet, allDataSet)
        return method
