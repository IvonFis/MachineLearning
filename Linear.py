"""We take the data from Salary Data, years of experience
vs salary. We will create a simple linear regression model."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LinearRegression

# we import the file
dataset = pd.read_csv("Salary_Data.csv")

# we create an arrey from the data
# we take all the colums except the last one
# n = the column with the dependent variable
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Split the dataset into the training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

"""LINEAR REGRESSION: We create an object from the class
We have some parameters for the inputs that are not obligatory."""
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

"""Plooting"""
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
