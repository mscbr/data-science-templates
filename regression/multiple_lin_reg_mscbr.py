#Multiple Linear Regression
# =============================================================================
# 
# =============================================================================
#ASSUMPTIONS:
#1. Linearity
#2. Homoscedasticity
#3. Multivariate normality
#4. Independence of errors
#5. Lack of multicollinearity


# Data Preprocessing Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding categorical data
 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3]) 
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray() 

# Avoiding the Dummy Var Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optional model using Backward elimination=======================
import statsmodels.formula.api as sm
#adding column b0x0
X = np.append(arr = np.ones((50, 1)).astype(int), values= X , axis = 1)
#matrix of optimal variables
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() #OLS - Optimal Least Square
regressor_OLS.summary()
#
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() #OLS - Optimal Least Square
regressor_OLS.summary()
#
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() #OLS - Optimal Least Square
regressor_OLS.summary()
#
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() #OLS - Optimal Least Square
regressor_OLS.summary()
#FIN
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() #OLS - Optimal Least Square
regressor_OLS.summary()
