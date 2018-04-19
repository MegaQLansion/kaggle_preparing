# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:39:22 2018

@author: Lansion
"""
#download data sets
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target

#split data sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#feature scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

#initialize training environment
from sklearn.neighbors import KNeighborsClassifier
KNC=KNeighborsClassifier()

#training with K Neighbors Classifier
KNC.fit(X_train,y_train)
KNC_y_predict=KNC.predict(X_test)
#rating
from sklearn.metrics import classification_report
print 'The accuracy of KNC is',KNC.score(X_test,y_test)
print 'The score of KNC is:\r',classification_report(y_test,KNC_y_predict,target_names=iris.target_names)