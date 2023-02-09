from copy import deepcopy
from Logger import MyLogger
from fileOperation import FileManager

class DataSet():
    def __init__(self, filename):
        self.filename = filename
        self.logger = MyLogger().getLogger()
        self.fileManager = FileManager()
        self.data = self.fileManager.readData(filename)
        self.logger.info("Create a new DataSet Object for {} data set".format(filename))

    def clone(self):
        self.logger.info("Clone {} Data set".format(self.filename))
        return deepcopy(self)

    def getDataSet(self):
        return self.data



extract1 = DataSet("car.data")
extract2 = DataSet("ecoli.data")
extract3 = extract2.clone()
# c = extract.readData("car")
# c = extract.moveTargetToLastColumn(c)
# print(c)
# extract.splitDataToTrainAndSet()
# print(extract.trainSet)
# print(extract.testSet)
