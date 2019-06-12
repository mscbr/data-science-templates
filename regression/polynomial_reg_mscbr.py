#Polynomial Regression mscbr

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #in order to have MATRIX not a vector
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set - NOT ENOUGH OBSERVATION FOR SPLIT & PURPOSES
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Reg (for comparison)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #degree=2 is default
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualising the Linear Regression results
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title('Truth / Bluff (Linear)')
plt.xlabel('Position Lvl')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results - USEFULL CODE
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1)) #<-- for HIGHER RESOLUTION
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth / Bluff (Linear)')
plt.xlabel('Position Lvl')
plt.ylabel('Salary')
plt.show()

#Prediction w/ Linear Reg
lin_reg.predict([[6.5]])


#Prediction w/ Polynomial Reg
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))