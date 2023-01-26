from copy import copy, deepcopy
from zope.interface import implementer
from AI_Algorithm import AIAlgorithmInterface
from src.Logger import MyLogger


@implementer(AIAlgorithmInterface)
class NaiveBayes:
    def __init__(self, data):
        self.logger = MyLogger().getLogger()
        self.outputMap = {}
        self.attributesMap = {}
        # Split Input and output
        self.output_Array = []
        self.input_Array = []
        for row in data:
            vector = deepcopy(row)
            self.output_Array.append(vector.pop())
            self.input_Array.append(vector)

    def readColumn(self, data, index):
        s = []
        for row in data:
            s.append(row[index])
        return s

    def train(self):
        # Validation
        if len(self.output_Array) != len(self.input_Array):
            self.logger.debug("Wrong Params , Check Prams")
            raise Exception("Wrong Params , Check Prams")
        # Discover output
        s = list(set(self.output_Array))
        # Count the output
        for category in self.output_Array:
            self.outputMap[category] = {"Count": self.outputMap.get(category, {"Count": 0}).get("Count") + 1}
        for category in self.outputMap:
            self.outputMap[category]["Probability"] = self.outputMap[category]["Count"] / len(self.output_Array)
        # Define Json for attributes
        #  Init Stage
        Number_Of_Columns = len(self.input_Array[0])
        for column in range(Number_Of_Columns):
            categories = set(self.readColumn(self.input_Array, column))
            self.attributesMap["COLUMN" + str(column)] = {}
            for value in categories:
                self.attributesMap["COLUMN" + str(column)][value] = {}
                for output_Category in self.outputMap:
                    self.attributesMap["COLUMN" + str(column)][value][output_Category] = 0
        #  Fill Stage
        for column in range(Number_Of_Columns):
            categories = self.readColumn(self.input_Array, column)
            POINTER = 0
            for value in categories:
                self.attributesMap["COLUMN" + str(column)][value][self.output_Array[POINTER]] += (
                            1 / self.outputMap[self.output_Array[POINTER]]["Count"])
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
        self.logger.info("Extract from model {} and the result equal {}".format(vector, result))
        return result
    

Data = [["Sunny", "Hot", "High", "Weak", "No"],
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
          ]
NB = NaiveBayes(Data)
NB.train()
print("Output:", NB.outputMap)
print("Attribute MAP:", NB.attributesMap)
res = NB.extractFromModel(["Overcast", "Cool", "Normal", "Strong"])
print(res)