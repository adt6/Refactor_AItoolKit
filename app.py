# Client Code ...
# This code need to be inside AIFactory
from __future__ import annotations
from src.AI_Factory import AIFactory
from src.CrossValidation import CrossValidation
from src.DataSet import DataSet
from src.ProductsCreator import ID3BoostCreator
from src.ProductsCreator import NaiveBayesCreator
from src.Logger import MyLogger
import sys
import statistics


from src.fileOperation import FileManager


if __name__ == "__main__":
    Ex = [["Sunny", "Hot", "High", "Weak", "No"],
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
    Data = FileManager("mushroom.data")
    dataSet = DataSet(Data.getDataSet())
    CrossValid = CrossValidation(4, 2, DataSet(Ex), "naivebayes")
    CrossValid.dataShufflingDT()
    result = CrossValid.triggerCrossValidation()
    average_Accuracy = sum(result) / len(result)
    std_dev = statistics.stdev(result)
    print("Standard Deviation: {}".format(std_dev))
    print("Average Accuracy: {}".format(average_Accuracy))

    """
    logger = MyLogger().getLogger()
    dataset_Name = sys.argv[1]
    Algorithm_Name = sys.argv[2]
    logger.info("Data Base name {} , Algorithm name {}".format(dataset_Name, Algorithm_Name))
    dataSET = DataSet(filename=dataset_Name)
    match Algorithm_Name:
        case "id3boost":
            AImodel = ID3BoostCreator(ID3BoostCreator())
        case "naivebayes":
            AImodel = createAIModel(NaiveBayesCreator())
        case _:
            print("This algorithm not supported yet")
            sys.exit(0)

    trainedModel = AImodel.createAIMethod(dataSET.getDataSet())
    Ex = [["Sunny", "Hot", "High", "Weak", "No"],
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
        DT = ID3BoostCreator()
    DT = DT.getAIMethod(Ex,Ex)
    DT.train()
    for row in Ex:
        print(row[-1], DT.extractFromModel(row))
    print("\n Start New Algorithm")
    DT = NaiveBayesCreator();
    DT = DT.getAIMethod(Ex,Ex)
    DT.train()
    for row in Ex:
        print(row[-1], DT.extractFromModel(row))
    """








