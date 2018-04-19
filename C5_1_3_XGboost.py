# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:38:13 2018

@author: Lansion
"""

import pandas as pd

titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

X=titanic[['pclass','age','sex']]
y=titanic['survived']

#complete age
X['age'].fillna(X['age'].mean(),inplace=True)
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#initialization
from sklearn.feature_extraction import DictVectorizer
DV=DictVectorizer(sparse=False)
#Vectorization
X_train=DV.fit_transform(X_train.to_dict(orient='record'))
X_test=DV.transform(X_test.to_dict(orient='record'))
#RandomForest
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier()
RFC.fit(X_train,y_train)
print 'Accuracy of RandomForest is',RFC.score(X_test,y_test)

#XGboost
from xgboost import XGBClassifier
xgbc=XGBClassifier()
xgbc.fit(X_train,y_train)
print 'Accuracy of XGBoost is',xgbc.score(X_test,y_test)