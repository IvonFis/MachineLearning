"""Basic template for Data Preprocessing for Machine Learning"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#AVOIDING THE DUMMY VARIABLE TRAP
X = X[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#CREATE AN OBJECT OF THE CLASS FOR THE LINEAR REGRETION
regressor= LinearRegression()
# FIT THE MULTIPLE LINEAR REGRESSOR TO THE TRAINING SET
regressor.fit(X_train, Y_train)

#PREDICTING THE TEST RESULTS, IS THE SAME AS LINEAR REGRESSION
Y_pred = regressor.predict(X_test)

"""
# BUILDING THE OPTIMAL MODEL USING BACKWARD ELIMINATION
# add the independent variable
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
#create an optimal matrix of features
# WE FIRST TAKE ALL THE COLUMS
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
a = regressor_OLS.summary()
print(a)
"""
"""OR
BACKWARD ELIMINATION WITH P VALUES ONLY

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    b = regressor_OLS.summary()
    print(b)
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""
"""
BACKWARD ELIMINATION WITH P VALUES AND ADJUSTED R SQUARED.
"""
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(Y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    print("begin")
                    print(adjR_after)
                    print("ends")
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    print(regressor_OLS.summary())
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
