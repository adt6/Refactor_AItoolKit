from src.AI_Algorithm import AIAlgorithmInterface
from src.Logger import MyLogger
from src.AI_Factory import AIFactory
from src.NaiveBayes import NaiveBayes
from src.AdaBoost import AdaBoost


class NaiveBayesCreator(AIFactory):
    def __init__(self):
        self.logger = MyLogger().getLogger()
        self.logger.info("Create Naive Bayes class creator")

    def createAIMethod(self, trainingDataSet, allDataSet) -> AIAlgorithmInterface:
        return NaiveBayes(trainingDataSet, allDataSet)


class ID3BoostCreator(AIFactory):
    def __init__(self):
        self.logger = MyLogger().getLogger()
        self.logger.info("Create Ada boost class creator")

    def createAIMethod(self, trainingDataSet, allDataSet) -> AIAlgorithmInterface:
        return AdaBoost(trainingDataSet, allDataSet)


