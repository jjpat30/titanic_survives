# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:10:22 2020

@author: jjpat
"""

#import libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read csv file
dataset= pd.read_csv('train.csv')
X_test=pd.read_csv('test.csv')
g_s=pd.read_csv('gender_submission.csv')
#X_train=dataset.iloc[:,2:]
#y_train=dataset.iloc[:,1].values

y_test=g_s.iloc[:,1].values

dataset=dataset.drop(['Name','Age','Fare','Ticket','Cabin'],axis=1)
X_test=X_test.drop(['PassengerId','Name','Age','Fare','Ticket','Cabin'],axis=1)
dataset= dataset.dropna()

X_train=dataset.iloc[:,2:]
y_train=dataset.iloc[:,1].values

X_train=X_train.values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le1=LabelEncoder()
X_train[:,1]=le1.fit_transform(X_train[:,1])
le2=LabelEncoder()
X_train[:,4]=le2.fit_transform(X_train[:,4])
onehotencoder = OneHotEncoder(categorical_features = [4])
X_train= onehotencoder.fit_transform(X_train).toarray()

# X_train  encodng categorical data
X_test=X_test.values
le3=LabelEncoder()
X_test[:,1]=le1.fit_transform(X_test[:,1])
le4=LabelEncoder()
X_test[:,4]=le2.fit_transform(X_test[:,4])
onehotencoder1 = OneHotEncoder(categorical_features = [4])
X_test= onehotencoder.fit_transform(X_test).toarray()

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#Predict the result
y_pred=classifier.predict(X_test)


# Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

np.savetxt('Titanic_survive_prediction', y_pred, delimiter=',',fmt='%0.0f') 
