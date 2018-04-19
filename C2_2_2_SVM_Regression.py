# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:21:33 2018

@author: Lansion
"""


#downloading initial data sets
from sklearn.datasets import load_digits
digits=load_digits()
X=digits.data
y=digits.target

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

#initialize training environment(with three kernels)
from sklearn.svm import SVR
LSVR=SVR(kernel='linear')
PSVR=SVR(kernel='poly')
SSVR=SVR(kernel='sigmoid')#classifier
RSVR=SVR(kernel='rbf')

#training with SVM
LSVR.fit(X_train,y_train)
LSVR_y_predict=LSVR.predict(X_test)
PSVR.fit(X_train,y_train)
PSVR_y_predict=PSVR.predict(X_test)
SSVR.fit(X_train,y_train)
SSVR_y_predict=SSVR.predict(X_test)
RSVR.fit(X_train,y_train)
RSVR_y_predict=RSVR.predict(X_test)

#rating
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'The R2 score of LSVR is:',r2_score(y_test,LSVR_y_predict)
print 'The R2 score of PSVR is:',r2_score(y_test,PSVR_y_predict)
print 'The R2 score of SSVR is:',r2_score(y_test,SSVR_y_predict)
print 'The R2 score of RSVR is:',r2_score(y_test,RSVR_y_predict)

print 'The mean squared error score of LSVR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(LSVR_y_predict))
print 'The mean squared error score of PSVR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(PSVR_y_predict))
print 'The mean squared error score of SSVR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(SSVR_y_predict))
print 'The mean squared error score of RSVR is:',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(RSVR_y_predict))

print 'The mean absolute error score of LSVR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(LSVR_y_predict))
print 'The mean absolute error score of PSVR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(PSVR_y_predict))
print 'The mean absolute error score of SSVR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(SSVR_y_predict))
print 'The mean absolute error score of RSVR is:',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(RSVR_y_predict))