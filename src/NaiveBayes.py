from __future__ import annotations
from copy import copy, deepcopy
from abc import ABC, abstractmethod
from src.AI_Algorithm import AIAlgorithmInterface
from src.Logger import MyLogger
from copy import copy, deepcopy



class NaiveBayes(AIAlgorithmInterface):
    def __init__(self, trainingDataSet, allDataSet=None):
        self.outputMap = {}
        self.attributesMap = {}
        # Split Input and output
        self.TrainingData_Output_Arr = []
        self.TrainingData_Input_Arr = []

        for row in trainingDataSet:
            vector = deepcopy(row)
            self.TrainingData_Output_Arr.append(vector.pop())
            self.TrainingData_Input_Arr.append(vector)
        # self.train()

    def _readColumn(self, data, index):
        s = []
        for row in data:
            s.append(row[index])
        return s

    def train(self):

        s = list(set(self.TrainingData_Output_Arr))
        # Count the output
        for category in self.TrainingData_Output_Arr:
            self.outputMap[category] = {"Count": self.outputMap.get(category, {"Count": 0}).get("Count") + 1}
        for category in self.outputMap:
            self.outputMap[category]["Probability"] = self.outputMap[category]["Count"] / len(self.TrainingData_Output_Arr)
        # Define Json for attributes
        #  Init Stage
        Number_Of_Columns = len(self.TrainingData_Input_Arr[0])
        for column in range(Number_Of_Columns):
            categories = set(self._readColumn(self.TrainingData_Input_Arr, column))
            self.attributesMap["COLUMN" + str(column)] = {}
            for value in categories:
                self.attributesMap["COLUMN" + str(column)][value] = {}
                for output_Category in self.outputMap:
                    self.attributesMap["COLUMN" + str(column)][value][output_Category] = 0
        #  Fill Stage
        for column in range(Number_Of_Columns):
            categories = self._readColumn(self.TrainingData_Input_Arr, column)
            POINTER = 0
            for value in categories:
                self.attributesMap["COLUMN" + str(column)][value][self.TrainingData_Output_Arr[POINTER]] += (
                            1 / self.outputMap[self.TrainingData_Output_Arr[POINTER]]["Count"])
                POINTER += 1

    def extractFromModel(self, vector):
        examine_result = {}
        for category in self.outputMap:
            examine_result[category] = 1

        # Calculate Prob for each Output
        for category in self.outputMap:
            examine_result[category] *= self.outputMap[category]["Probability"]
            for c in range(len(vector)):
                try:
                    examine_result[category] *= self.attributesMap["COLUMN"+str(c)][vector[c]][category]
                except KeyError:
                    examine_result[category] *= 0
        result = sorted(examine_result, key=lambda x: examine_result[x])[-1]
        print("Result {}".format(result))
        return result
"""
NB = NaiveBayes([["Sunny", "Hot", "High", "Weak", "No"],
          ["Sunny", "Hot", "High", "Strong", "No"],
          ["Overcast", "Hot", "High", "Weak", "Yes"],
          ["Rain", "Mild", "High", "Weak", "Yes"],
          ["Rain", "Cool", "Normal", "Weak", "Yes"],
          ["Rain", "Cool", "Normal", "Strong", "No"],
          ["Overcast", "Cool", "Normal", "Strong", "Yes"],
          ["Sunny", "Mild", "High", "Weak", "No"],
          ["Sunny", "Cool", "Normal", "Weak", "Yes"],
          ["Rain", "Mild", "Normal", "Weak", "Yes"],
          ["Sunny", "Mild", "Normal", "Strong", "Yes"],
          ["Overcast", "Mild", "High", "Strong", "Yes"],
          ["Overcast", "Hot", "Normal", "Weak", "Yes"],
          ["Rain", "Mild", "High", "Strong", "No"]
          ])
NB.train()
NB.extractFromModel(["Sunny", "Cool", "High", "Strong"])


"""












