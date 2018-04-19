# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:52:51 2018

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

from sklearn.linear_model import LinearRegression,SGDRegressor 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
LR=LinearRegression()
SGDR=SGDRegressor()
LSVR=SVR(kernel='linear')
PSVR=SVR(kernel='poly')
RSVR=SVR(kernel='rbf')
uni_KNR=KNeighborsRegressor(weights='uniform')
dis_KNR=KNeighborsRegressor(weights='distance')
DTR=DecisionTreeRegressor()
RFR=RandomForestRegressor()
ETR=ExtraTreesRegressor()
GBR=GradientBoostingRegressor()

#training & prediction
LR.fit(X_train,y_train)
LR_y_predict=LR.predict(X_test)
SGDR.fit(X_train,y_train)
SGDR_y_predict=SGDR.predict(X_test)
LSVR.fit(X_train,y_train)
LSVR_y_predict=LSVR.predict(X_test)
PSVR.fit(X_train,y_train)
PSVR_y_predict=PSVR.predict(X_test)
RSVR.fit(X_train,y_train)
RSVR_y_predict=RSVR.predict(X_test)
uni_KNR.fit(X_train,y_train)
uni_KNR_y_predict=uni_KNR.predict(X_test)
dis_KNR.fit(X_train,y_train)
dis_KNR_y_predict=dis_KNR.predict(X_test)
DTR.fit(X_train,y_train)
DTR_y_predict=DTR.predict(X_test)
RFR.fit(X_train,y_train)
ETR.fit(X_train,y_train)
GBR.fit(X_train,y_train)
RFR_y_predict=RFR.predict(X_test)
ETR_y_predict=ETR.predict(X_test)
GBR_y_predict=GBR.predict(X_test)

#rating
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'The R2 score of LR is:',r2_score(y_test,LR_y_predict)
print 'The R2 score of SGDR is:',r2_score(y_test,SGDR_y_predict)
print 'The R2 score of LSVR is:',r2_score(y_test,LSVR_y_predict)
print 'The R2 score of PSVR is:',r2_score(y_test,PSVR_y_predict)
print 'The R2 score of RSVR is:',r2_score(y_test,RSVR_y_predict)
print 'The R2 score of uni_KNR is:',r2_score(y_test,uni_KNR_y_predict)
print 'The R2 score of dis_KNR is:',r2_score(y_test,dis_KNR_y_predict)
print 'The R2 score of DTR is:',r2_score(y_test,DTR_y_predict)
print 'The R2 score of RFR is:',r2_score(y_test,RFR_y_predict)
print 'The R2 score of ETR is:',r2_score(y_test,ETR_y_predict)
print 'The R2 score of GBR is:',r2_score(y_test,GBR_y_predict)

print 'The MQE score of LR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(LR_y_predict))
print 'The MQE score of SGDR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(SGDR_y_predict))
print 'The MQE score of LSVR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(LSVR_y_predict))
print 'The MQE score of PSVR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(PSVR_y_predict))
print 'The MQE score of RSVR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(RSVR_y_predict))
print 'The MQE score of uni_KNR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_KNR_y_predict))
print 'The MQE score of dis_KNR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_KNR_y_predict))
print 'The MQE score of DTR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(DTR_y_predict))
print 'The MQE score of RFR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(RFR_y_predict))
print 'The MQE score of ETR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(ETR_y_predict))
print 'The MQE score of GBR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(GBR_y_predict))

print 'The MAE score of LR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(LR_y_predict))
print 'The MAE score of SGDR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(SGDR_y_predict))
print 'The MAE score of LSVR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(LSVR_y_predict))
print 'The MAE score of PSVR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(PSVR_y_predict))
print 'The MAE score of RSVR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(RSVR_y_predict))
print 'The MAE score of uni_KNR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_KNR_y_predict))
print 'The MAE score of dis_KNR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_KNR_y_predict))
print 'The MAE score of DTR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(DTR_y_predict))
print 'The MAE score of RFR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(RFR_y_predict))
print 'The MAE score of ETR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(ETR_y_predict))
print 'The MAE score of GBR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(GBR_y_predict))

import numpy as np
feature_importance=np.sort(zip(RFR.feature_importances_,boston.feature_names),axis=0)
most_imp[0]=feature_importance[feature_importance[:,0]==max(feature_importance[:,0])]
print 'The most important feature is',most_imp[0,1],'with relavant parameter',most_imp[0,0]