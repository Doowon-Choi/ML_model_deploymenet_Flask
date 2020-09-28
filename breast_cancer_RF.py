# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:10:03 2020

@author: Doowon
"""

%matplotlib inline

import numpy as np
import pandas as pd
import seaborn as s
import pickle
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

##C:\Users\Doowon\Documents\Model_deploy\all\predict_breast\demo
data = pd.read_csv('breast_data.csv')
df = data.drop('Unnamed: 32', axis=1) #####remove the column

## check the degree of balance
df['diagnosis'].value_counts()

## make the response as categorical variable
df.diagnosis = df.diagnosis.astype('category')

### select X and Y
X = df.drop(labels='diagnosis', axis=1)
X = X.drop(labels = 'id', axis = 1)
Y = df['diagnosis']
col = X.columns

### Feature engineering
X.isnull().sum()

#### normalization
X.head()
X_norm = (X - X.mean()) / (X.max()-X.min())
df_norm = pd.concat([X_norm, Y], axis = 1)
Y_norm = df_norm['diagnosis']

col = X_norm.columns

#### transform level in Y into number
le =  LabelEncoder()### from sklearn.preprocessing
le.fit(Y_norm)
Y_norm = le.transform(Y_norm) ### (569,) for passing to the ML model

############## Fitting classification models ##########
x_train, x_test, y_train, y_test = train_test_split(X_norm, Y_norm, test_size = 0.2)
param = { 'n_estimators': [100, 500, 1000, 1500]}

grid = GridSearchCV(
        estimator = RandomForestClassifier(),
        param_grid = param ,cv = 5,
        scoring = 'accuracy', verbose = 1, n_jobs = -1)
    
grid_result = grid.fit(x_train, y_train)
best_params = grid_result.best_params_
pred = grid_result.predict(x_test.iloc[1])
pred = grid_result.predict(asp.reshape(1,30))

cm = confusion_matrix(y_test, pred)

#### save object here object is ML model
pickle.dump(grid_result, open('RF.pkl','wb')) ### for creating and saving ML model

print('Best Params :', best_params)
print('Classification Report :', classification_report(y_test, pred))
print('Accuracy Score : ' + str(accuracy_score(y_test, pred)))
print('Confusion Matrix : \n', cm)
   
### reload saved model
loaded_model = pickle.load(open("RF.pkl","rb"))
pred_value = loaded_model.predict(x_test)
loaded_model.best_params_