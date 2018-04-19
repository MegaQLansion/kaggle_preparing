# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:39:59 2018

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
ss_X=StandardScaler()
ss_y=StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train.reshape(-1,1))
y_test=ss_y.transform(y_test.reshape(-1,1))

#initialize training environment
from sklearn.neighbors import KNeighborsRegressor
uni_KNR=KNeighborsRegressor(weights='uniform')
dis_KNR=KNeighborsRegressor(weights='distance')

#training with K Neighbors Regressor
uni_KNR.fit(X_train,y_train)
uni_KNR_y_predict=uni_KNR.predict(X_test)
dis_KNR.fit(X_train,y_train)
dis_KNR_y_predict=dis_KNR.predict(X_test)

#rating
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'The R2 score of uni_KNR is:',r2_score(y_test,uni_KNR_y_predict)
print 'The R2 score of dis_KNR is:',r2_score(y_test,dis_KNR_y_predict)


print 'The mean squared error score of uni_KNR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_KNR_y_predict))
print 'The mean squared error score of dis_KNR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_KNR_y_predict))


print 'The mean absolute error score of uni_KNR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_KNR_y_predict))
print 'The mean absolute error score of dis_KNR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_KNR_y_predict))
