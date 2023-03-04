from random import random

from src.Logger import MyLogger
from copy import deepcopy
import numpy as np

class FileManager:
    def __init__(self, dataset_Name):
        self.logger = MyLogger().getLogger()
        # working data set: car,mushroom,ecoli
        data = None
        data = self.readData(dataset_Name)
        # Because the target in  mushroom Data set and Letter data set is in the 1st column
        if dataset_Name == "mushroom":
            data = self.moveTargetToLastColumn(data)
        if dataset_Name == "letter":
            data = self.moveTargetToLastColumn(data)
            # for each column convert continuous to Discrete :5,6
            columnIndexToChange = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            for index in columnIndexToChange:
                array = self.readColumn(data, index)
                Discreate_array = self.ConvertRowToDiscreate(array, 15)
                self.SaveColumn(data, Discreate_array, index)
        if dataset_Name == "ecoli":
            # Remove 1st column
            data = self.deleteColumn(data, 0)
            # for each column convert continuous to Discrete :5,6
            columnIndexToChange = [0, 1, 2, 3, 4, 5, 6]
            for index in columnIndexToChange:
                array = self.readColumn(data, index)
                Discreate_array = self.ConvertRowToDiscreate(array)
                self.SaveColumn(data, Discreate_array, index)
        if dataset_Name == "cancer":
            data = self.deleteColumn(data, 0)
            # for each column convert continuous to Discrete :5,6
            columnIndexToChange = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            for index in columnIndexToChange:
                array = self.readColumn(data, index)
                Discreate_array = self.ConvertRowToDiscreate(array)
                self.SaveColumn(data, Discreate_array, index)
        self.DataSet =  data
        # random.shuffle(data)
        # accuracy = dataShufflingDT(data)
        # print("Accuracy: ", sum(accuracy) / 50.0)
        # print("Standard Deviation: ", statistics.stdev(accuracy))
    def getDataSet(self):
        return self.DataSet

    def readData(self, fileName):
        Dataset = open('./DataSet_DataBase/' + fileName, 'r')
        data = Dataset.readlines()
        Array = []
        T = ","
        # ecoli Data set separated by double empty space
        if fileName == "ecoli":
            T = "  "
        for line in data:
            # remove \n char from the end of the text ,
            # then split the row depends on comma
            record = line[:-1].split(T)
            if '?' not in record: Array.append(record)
        self.logger.info(
            "Reading Data set:{} Feature Numbers:{} Number of rows:{}".format(fileName, len(Array[0]), len(Array)))
        return Array

    def moveTargetToLastColumn(self, data):
        modifyData = deepcopy(data)
        row = len(modifyData)
        column = len(modifyData[0])
        for i in range(row):
            temp = modifyData[i][0]
            modifyData[i][0] = modifyData[i][column - 1]
            modifyData[i][column - 1] = temp
        self.logger.info("Move first column to last column")
        return modifyData

    def deleteColumn(self, data, index):
        modifyData = deepcopy(data)
        for row in modifyData:
            del row[index]
        self.logger.info("Delete index {} from data set".format(index))
        return modifyData

    def readColumn(self, data, index):
        s = []
        for row in data:
            s.append(row[index])
        return s

    def ConvertRowToDiscreate(self, InputRow, categoryInterval=5):
        Row = [float(x) for x in InputRow]
        maxValue = max(Row)
        minValue = min(Row)
        # we will always divide it to 5 category
        distance = (maxValue - minValue) / categoryInterval
        print("Distance", distance)
        ruler = [minValue]
        while ruler[-1] < maxValue:
            ruler.append(ruler[-1] + distance)
        print("Ruler:", ruler)
        discreateArray = []
        for val in Row:
            # Check Ruler to find the interval
            for i in range(len(ruler)):
                if i == len(ruler) - 1:
                    discreateArray.append(str(i - 1))
                    break
                elif ruler[i] <= val < ruler[i + 1]:
                    discreateArray.append(str(i))
                    break
        # Check the length
        if len(Row) != len(discreateArray):
            raise Exception("Error in Converting Numeric value to categorial values")
        return discreateArray

    def SaveColumn(self, OriginArray, ColumnData, index):
        count = 0
        for value in ColumnData:
            OriginArray[count][index] = value
            count += 1