"""Basic template for Polynomial Regression for Machine Learning"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# we import the file
dataset = pd.read_csv("filename.csv")

# we create an arrey from the data
# we take all the colums except the last one
# n = the column with the dependent variable
"""
n = 3
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, n].values
"""
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values


# Split the dataset into the training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# OPTIONAL: Feature scaling: all the variables will have the same range
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# FITTING THE REGRESSION MODEL


#PREDICTING THE RESULT 
print(Y_pred = regressor.predict(6.5))

# PLOTTING
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Salary vs Experience Polynomial')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
