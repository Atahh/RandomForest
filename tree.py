import numpy as np
import pandas as pd
from informationGain import information_gain

"""
Load data loads the training data
"""
def load_TrainData():
    trainData = pd.read_csv("train.csv",sep=',',header=None,skiprows=1)
    #print("Data Length: ", len(trainData))
    #print("Data Shape: ", trainData.shape)
    #print("Data: ", trainData.head)
    
    return trainData

def load_TestData():
    testData = pd.read_csv("test.csv",sep=',',header=None,skiprows=1)
    return testData

def splitdataset(balance_data):
 
    # Separating the target variable
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]
 
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)
 
    return X, Y, X_train, X_test, y_train, y_test


    