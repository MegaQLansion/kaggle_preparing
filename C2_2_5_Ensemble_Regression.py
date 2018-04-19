# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:25:09 2018

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
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
RFR=RandomForestRegressor()
ETR=ExtraTreesRegressor()
GBR=GradientBoostingRegressor()
#Training
RFR.fit(X_train,y_train)
ETR.fit(X_train,y_train)
GBR.fit(X_train,y_train)
#prediction
RFR_y_predict=RFR.predict(X_test)
ETR_y_predict=ETR.predict(X_test)
GBR_y_predict=GBR.predict(X_test)

#rating
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'The R2 score of RFR is:',r2_score(y_test,RFR_y_predict)
print 'The R2 score of ETR is:',r2_score(y_test,ETR_y_predict)
print 'The R2 score of GBR is:',r2_score(y_test,GBR_y_predict)

print 'The mean squared error score of RFR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(RFR_y_predict))
print 'The mean squared error score of ETR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(ETR_y_predict))
print 'The mean squared error score of GBR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(GBR_y_predict))

print 'The mean absolute error score of RFR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(RFR_y_predict))
print 'The mean absolute error score of ETR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(ETR_y_predict))
print 'The mean absolute error score of GBR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(GBR_y_predict))
#feature importance
import numpy as np
feature_importance=np.sort(zip(ETR.feature_importances_,boston.feature_names),axis=0)
most_imp[0]=feature_importance[feature_importance[:,0]==max(feature_importance[:,0])]
print 'The most important feature is',most_imp[0,1],'with relavant parameter',most_imp[0,0]