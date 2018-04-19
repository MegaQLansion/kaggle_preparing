# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:45:45 2018

@author: Lansion
"""
#preprocessing of data sets
from sklearn.datasets import load_boston
boston=load_boston()
print 'Data sets description:\r',boston.DESCR
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

#initialization
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.linear_model import SGDRegressor
SGDR=SGDRegressor()


#training with lr
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)

#training with SGD
SGDR.fit(X_train,y_train)
SGDR_y_predict=SGDR.predict(X_test)


#rating
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'The R2 score of LR is:',r2_score(y_test,lr_y_predict)
print 'The R2 score of SGDR is:',r2_score(y_test,SGDR_y_predict)
print 'The mean squared error score of LR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))
print 'The mean squared error score of SGDR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(SGDR_y_predict))
print 'The mean absolute error score of LR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))
print 'The mean absolute error score of SGDR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(SGDR_y_predict))