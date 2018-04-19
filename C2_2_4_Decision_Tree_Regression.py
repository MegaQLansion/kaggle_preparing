# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:06:04 2018

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
from sklearn.tree import DecisionTreeRegressor
DTR=DecisionTreeRegressor()



#training with DTR
DTR.fit(X_train,y_train)
DTR_y_predict=DTR.predict(X_test)



#rating
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'The R2 score of DTR is:',r2_score(y_test,DTR_y_predict)
print 'The mean squared error score of DTR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(DTR_y_predict))
print 'The mean absolute error score of DTR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(DTR_y_predict))
