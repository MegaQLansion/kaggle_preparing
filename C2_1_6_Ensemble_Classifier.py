# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:37:20 2018

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
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier()
from sklearn.ensemble import GradientBoostingClassifier
GBDT=GradientBoostingClassifier()
#training with decision tree
DTC.fit(X_train,y_train)
DTC_y_predict=DTC.predict(X_test)

#rating
from sklearn.metrics import classification_report
print 'The accuracy of decition tree is',DTC.score(X_test,y_test)
print 'The score of Decition Tree is:\r',classification_report(y_test,DTC_y_predict,target_names=['Die','Survive'])


#training with random forest
RFC.fit(X_train,y_train)
RFC_y_predict=RFC.predict(X_test)

#rating
print 'The accuracy of Random Forest is',RFC.score(X_test,y_test)
print 'The score of Random Forest is:\r',classification_report(y_test,RFC_y_predict,target_names=['Die','Survive'])


#training with GBDT
GBDT.fit(X_train,y_train)
GBDT_y_predict=GBDT.predict(X_test)

#rating
print 'The accuracy of GBDT is',GBDT.score(X_test,y_test)
print 'The score of GBDT is:\r',classification_report(y_test,GBDT_y_predict,target_names=['Die','Survive'])

