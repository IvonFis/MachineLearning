import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("Position_Salaries.csv")
# remember to consider X as a matrix and not as a vector
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# LINEAR REGRESSION MODEL
regressor_lin = LinearRegression()
regressor_lin.fit(X, Y)

# POLYNOMIAL REGRESSION
regressor_pol = PolynomialFeatures(degree = 4)
X_pol = regressor_pol.fit_transform(X)
regressor_pol.fit(X_pol, Y)
regressor_lin2 = LinearRegression()
regressor_lin2.fit(X_pol, Y)

"""Plooting linear
plt.scatter(X, Y, color='red')
plt.plot(X, regressor_lin.predict(X), color='blue')
plt.title('Salary vs Experience Linear')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
"""

"""Plooting polynomial"""
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor_lin2.predict(regressor_pol.fit_transform(X_grid)), color='blue')
plt.title('Salary vs Experience Polynomial')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#PREDICTING THE RESULT LINEAR AND POLYNOMIAL
print(regressor_lin.predict(6.5))
print(regressor_lin2.predict(regressor_pol.fit_transform(6.5)))
