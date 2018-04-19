# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 18:37:36 2018

@author: Lansion
"""


import pandas as pd
titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
X=titanic[['pclass','age','sex']]
y=titanic['survived']

#complete age using average value
X['age'].fillna(X['age'].mean(),inplace=True)

#split data sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)
#feature extraction
from sklearn.feature_extraction import DictVectorizer
DV=DictVectorizer(sparse=False)
X_train=DV.fit_transform(X_train.to_dict(orient='record'))
X_test=DV.transform(X_test.to_dict(orient='recored'))


#initialize training environment
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import NuSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
xgbc=XGBClassifier()
LR=LogisticRegression()
SGDC=SGDClassifier()
Linear_SVC=NuSVC(kernel='linear')
Poly_SVC=NuSVC(kernel='poly')
Rbf_SVC=NuSVC(kernel='rbf')
Sigmoid_SVC=NuSVC(kernel='sigmoid')
MNB=MultinomialNB()
KNC=KNeighborsClassifier()
DTC=DecisionTreeClassifier()
RFC=RandomForestClassifier()
GBDT=GradientBoostingClassifier()

#training
LR.fit(X_train,y_train)
LR_y_predict=LR.predict(X_test)
SGDC.fit(X_train,y_train)
SGDC_y_predict=SGDC.predict(X_test)
Linear_SVC.fit(X_train,y_train)
Linear_SVC_y_predict=Linear_SVC.predict(X_test)
Poly_SVC.fit(X_train,y_train)
Poly_SVC_y_predict=Poly_SVC.predict(X_test)
Rbf_SVC.fit(X_train,y_train)
Rbf_SVC_y_predict=Rbf_SVC.predict(X_test)
Sigmoid_SVC.fit(X_train,y_train)
Sigmoid_SVC_y_predict=Sigmoid_SVC.predict(X_test)
MNB.fit(X_train,y_train)
MNB_y_predict=MNB.predict(X_test)
KNC.fit(X_train,y_train)
KNC_y_predict=KNC.predict(X_test)
DTC.fit(X_train,y_train)
DTC_y_predict=DTC.predict(X_test)
RFC.fit(X_train,y_train)
RFC_y_predict=RFC.predict(X_test)
GBDT.fit(X_train,y_train)
GBDT_y_predict=GBDT.predict(X_test)
xgbc.fit(X_train,y_train)
xgbc_y_predict=xgbc.predict(X_test)
#rating
from sklearn.metrics import f1_score
print 'The accuracy of LR is:',LR.score(X_test,y_test)
print 'The accuracy of SGDClassifier is:',SGDC.score(X_test,y_test)
print 'The accuracy of Linear_SVC is:',Linear_SVC.score(X_test,y_test)
print 'The accuracy of Poly_SVC is:',Poly_SVC.score(X_test,y_test)
print 'The accuracy of Rbf_SVC is:',Rbf_SVC.score(X_test,y_test)
print 'The accuracy of Sigmoid_SVC is:',Sigmoid_SVC.score(X_test,y_test)
print 'The accuracy of Naive Bayes is:',MNB.score(X_test,y_test)
print 'The accuracy of KNC is:',KNC.score(X_test,y_test)
print 'The accuracy of decition tree is:',DTC.score(X_test,y_test)
print 'The accuracy of Random Forest is:',RFC.score(X_test,y_test)
print 'The accuracy of GBDT is:',GBDT.score(X_test,y_test)
print 'The accuracy of XGBoost is',xgbc.score(X_test,y_test)



print 'The F1-score of LR is:',f1_score(y_test,LR_y_predict)
print 'The F1-score of SGDClassifier is:',f1_score(y_test,SGDC_y_predict)
print 'The F1-score of Linear_SVC is:',f1_score(y_test,Linear_SVC_y_predict)
print 'The F1-score of Poly_SVC is:',f1_score(y_test,Poly_SVC_y_predict)
print 'The F1-score of Rbf_SVC is:',f1_score(y_test,Rbf_SVC_y_predict)
print 'The F1-score of Sigmoid_SVC is:',f1_score(y_test,Sigmoid_SVC_y_predict)
print 'The F1-score of Naive Bayes is:',f1_score(y_test,MNB_y_predict)
print 'The F1-score of KNC is:',f1_score(y_test,KNC_y_predict)
print 'The F1-score of decition tree is:',f1_score(y_test,DTC_y_predict)
print 'The F1-score of Random Forest is:',f1_score(y_test,RFC_y_predict)
print 'The F1-score of GBDT is:',f1_score(y_test,GBDT_y_predict)
print 'The F1-score of XGBoost is:',f1_score(y_test,xgbc_y_predict)