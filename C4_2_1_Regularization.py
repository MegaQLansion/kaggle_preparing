# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:56:53 2018

@author: Lansion
"""

X_train=[[6],[8],[10],[14],[18]]
y_train=[[7],[9],[13],[17.5],[18]]
X_test=[[6],[8],[11],[16]]
y_test=[[8],[12],[15],[18]]
from sklearn.linear_model import LinearRegression
#polynomial regression is indeed a feature preprocessing
from sklearn.preprocessing import PolynomialFeatures
poly4=PolynomialFeatures(degree=4)
X_train_poly4=poly4.fit_transform(X_train)
X_test_poly4=poly4.transform(X_test)

LR_poly4=LinearRegression()
LR_poly4.fit(X_train_poly4,y_train)
print 'The R2 for polynoial regression of degree 4 without Regularization is',LR_poly4.score(X_test_poly4,y_test)

#L1 Regularization
from sklearn.linear_model import Lasso
lasso_poly4=Lasso()
lasso_poly4.fit(X_train_poly4,y_train)
print 'The R2 for polynoial regression of degree 4 with L1 Regularization is',lasso_poly4.score(X_test_poly4,y_test)

#L2 Regularization
from sklearn.linear_model import Ridge
ridge_poly4=Ridge()
ridge_poly4.fit(X_train_poly4,y_train)
print 'The R2 for polynoial regression of degree 4 with L2 Regularization is',ridge_poly4.score(X_test_poly4,y_test)
