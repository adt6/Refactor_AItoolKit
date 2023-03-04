from copy import deepcopy

from src.DataSet import DataSet
from src.ProductsCreator import NaiveBayesCreator


class CrossValidation:
    def __init__(self, iteration, foldNumbers, DataSet, algorithmName):
        self.iteration = iteration
        self.foldNumbers = foldNumbers
        self._DataSet = DataSet
        self._generatedDataSets = []
        match algorithmName.lower():
            case "adaboost":
                from src.ProductsCreator import ID3BoostCreator
                self.AIModel = ID3BoostCreator()
            case "naivebayes":
                self.AIModel = NaiveBayesCreator()
            case _:
                self.AIModel = None

    def dataShufflingDT(self):
        sumOfAccuracy = []
        blockToShuffale = int(self._DataSet.getDataSetLength() / self.iteration)
        for i in range(self.iteration):
            print("Shuffle #: ", i)
            self._DataSet.shuffleDataByBlockSize(blockToShuffale)
            blockSize = int(self._DataSet.getDataSetLength() / self.foldNumbers)
            DataCopy = self._DataSet.clone()
            for c in range(self.foldNumbers):
                trainSet = []
                testSet = []
                DataCopy.shuffleDataByBlockSize(blockToShuffale)
                testSet = DataCopy.getDataFromTheBeginning(blockSize)
                trainSet = DataCopy.getDataFromTheEnd(blockSize)
                self._generatedDataSets.append({"trainSet": trainSet, "testSet": testSet})

    def triggerCrossValidation(self):
        if len(self._generatedDataSets) == 0:
            print("No generated data sets , please call dataShufflingDT function before calling this function")
            return
        if self.AIModel is None:
            print("AI model not supported")
            return
        # print("Data set length {}".format(len(self._generatedDataSets)))
        sumOfAccuracy = []
        for dataset in self._generatedDataSets:
            correctMatch = 0
            trainSet = dataset["trainSet"]
            testSet = dataset["testSet"]
            Model = self.AIModel.getAIMethod(trainSet+testSet, trainSet)
            Model.train()
            for record in range(len(testSet)):
                input_rec = deepcopy(testSet[record])
                input_rec.pop()
                result = Model.extractFromModel(input_rec)
                # print("Input record ", testSet[record], " : ", result)
                if result == testSet[record][-1]:
                    correctMatch += 1
            sumOfAccuracy.append(correctMatch / len(testSet) * 100.0)
        return sumOfAccuracy





