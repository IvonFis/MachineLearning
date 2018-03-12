# DATA PREPROCESING FOR ML
#HOW TO IMPORT THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split


# import the dataset
# we need to specify the working directory,
# the folder that containd the dataset

dataset = pd.read_csv("Data.csv")

# we need to create the matrix of features,
# creates an array
# we take all the lines, after the coma are the colums
# we take all the colums except the last one

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Take care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Categorical data
""" We need to encode the strings as values, we use the
dummy variables"""

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# WE SPLIT THE DATASET INTO THE TRAINING SET AND TEST TEST
# we create
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
