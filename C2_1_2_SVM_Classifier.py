# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:31:20 2018

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
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

#initialize training environment
from sklearn.svm import NuSVC
Linear_SVC=NuSVC(kernel='linear')
Poly_SVC=NuSVC(kernel='poly')
Rbf_SVC=NuSVC(kernel='rbf')
Sigmoid_SVC=NuSVC(kernel='sigmoid')

#training with SVM
Linear_SVC.fit(X_train,y_train)
Linear_SVC_y_predict=Linear_SVC.predict(X_test)
print 'The accuracy of Linear_SVC is:',Linear_SVC.score(X_test,y_test)

Poly_SVC.fit(X_train,y_train)
Poly_SVC_y_predict=Poly_SVC.predict(X_test)
print 'The accuracy of Poly_SVC is:',Poly_SVC.score(X_test,y_test)


Rbf_SVC.fit(X_train,y_train)
Rbf_SVC_y_predict=Rbf_SVC.predict(X_test)
print 'The accuracy of Rbf_SVC is:',Rbf_SVC.score(X_test,y_test)

Sigmoid_SVC.fit(X_train,y_train)
Sigmoid_SVC_y_predict=Sigmoid_SVC.predict(X_test)
print 'The accuracy of Sigmoid_SVC is:',Sigmoid_SVC.score(X_test,y_test)
