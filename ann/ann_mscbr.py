#ANN
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building ANN

#keras import
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing ANN
classifier = Sequential()
#adding input layer & the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11 ))
#adding 2nd layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
#adding output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=25)

# Predicting the Test set results
y_pred = classifier.predict(X_test, batch_size=10)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Predicting particular client

X_hw = np.array([[0,0,600, 1, 40,3,60000,2,1,1,50000]])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_hw = sc.fit_transform(X_hw)

y_pred_hw = classifier.predict(X_hw, batch_size=10)
y_pred_hw = (y_pred_hw > 0.5)

