# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:44:18 2018

@author: Lansion
"""

from sklearn import datasets,metrics,preprocessing,cross_validation
#preprocessing of data sets
from sklearn.datasets import load_boston
boston=load_boston()
print ('Data sets description:',boston.DESCR)
X=boston.data
y=boston.target

#split data sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#feature scaling
from sklearn.preprocessing import StandardScaler
ss_X=StandardScaler()
ss_y=StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train.reshape(-1,1))
y_test=ss_y.transform(y_test.reshape(-1,1))

#skflow
import skflow
tf_lr=skflow.TensorFlowLinearRegressor(steps=10000,learning_rate=0.01,batch_size=50)
tf_lr.fit(X_train,y_train)
tf_lr_y_predict=tf_lr.predict(X_test)

print ('The mean absoluate error of Tensorflow Linear Regressor on boston dataset is',metrics.mean_absolute_error(tf_lr_y_predict,y_test))
print ('The mean squared error of Tensorflow Linear Regressor on boston dataset is',metrics.mean_squared_error(tf_lr_y_predict,y_test))
