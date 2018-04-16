"""Basic template for Data Preprocessing for Machine Learning"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

# we import the file
dataset = pd.read_csv("filename.csv")

# we create an arrey from the data
# we take all the colums except the last one
# n = the column with the dependent variable
n = 3
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, n].values

# Split the dataset into the training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# OPTIONAL: Feature scaling: all the variables will have the same range
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
