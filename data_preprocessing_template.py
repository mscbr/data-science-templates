

#missing data
# =============================================================================
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values=np.nan, strategy="mean", axis=0)
# imputer = imputer.fit(X[:,1:3])
# X[:,1:3] = imputer.transform(X[:,1:3])
# =============================================================================

#encoding categorical data
# =============================================================================
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:,0] = labelencoder_X.fit_transform(X[:,0]) 
# onehotencoder = OneHotEncoder(categorical_features=[0])
# X = onehotencoder.fit_transform(X).toarray()
# 
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
# =============================================================================
#Data Preprocessing

#Libraries import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#splitting the data test vs. train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0) #random_state = 42 is a good choice

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #we only fit to X_train to have same scaling on X_test
X_test = sc_X.transform(X_test)