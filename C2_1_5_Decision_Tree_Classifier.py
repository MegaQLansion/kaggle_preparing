# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:50:23 2018

@author: Lansion
"""

import pandas as pd
titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
X=titanic[['pclass','age','sex']]
y=titanic['survived']

#complete age using average value
X['age'].fillna(X['age'].mean(),inplace=True)

#split data sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)
#feature extraction
from sklearn.feature_extraction import DictVectorizer
DV=DictVectorizer(sparse=False)
X_train=DV.fit_transform(X_train.to_dict(orient='record'))
X_test=DV.transform(X_test.to_dict(orient='recored'))
#initialization
from sklearn.tree import DecisionTreeClassifier
DTC=DecisionTreeClassifier()

#training with decision tree
DTC.fit(X_train,y_train)
DTC_y_predict=DTC.predict(X_test)

#rating
from sklearn.metrics import classification_report
print 'The accuracy of decition tree is',DTC.score(X_test,y_test)
print 'The score of Decition Tree is:\r',classification_report(y_test,DTC_y_predict,target_names=['Die','Survive'])