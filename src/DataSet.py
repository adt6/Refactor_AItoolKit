from copy import deepcopy
from src.Logger import MyLogger
from src.fileOperation import FileManager

class DataSet():
    def __init__(self, data):
        # self.filename = filename
        self.logger = MyLogger().getLogger()
        # self.fileManager = FileManager()
        # self.data = self.fileManager.readData(filename)
        self._data = data
        self.logger.info("Create a new DataSet Object for {} data set")

    def clone(self):
        self.logger.info("Clone Data set")
        return deepcopy(self)

    def getDataSet(self):
        return deepcopy(self._data)

    def shuffleDataByBlockSize(self, blockSize):
        for y in range(blockSize):
            temp = self._data.pop()
            self._data.insert(0, temp)

    def getDataSetLength(self):
        return len(self._data)

    def getDataFromTheBeginning(self, index):
        return self._data[:index]

    def getDataFromTheEnd(self, index):
        return self._data[index+1:]







# extract1 = DataSet("car.data")
# extract2 = DataSet("ecoli.data")
# extract3 = extract2.clone()
# c = extract.readData("car")
# c = extract.moveTargetToLastColumn(c)
# print(c)
# extract.splitDataToTrainAndSet()
# print(extract.trainSet)
# print(extract.testSet)
