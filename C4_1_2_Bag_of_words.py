# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:41:28 2018

@author: Lansion
"""

from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups(subset='all')

#split data sets
from sklearn.cross_validation import train_test_split
X=news.data
y=news.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#initialization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
CV=CountVectorizer()
TV=TfidfVectorizer()
MNB_CV=MultinomialNB()#naive bayes using CountVectorizer
MNB_TV=MultinomialNB()#naive bayes using TfidfVectorizer

#feature transforming
X_CV_train=CV.fit_transform(X_train)
X_CV_test=CV.transform(X_test)
X_TV_train=TV.fit_transform(X_train)
X_TV_test=TV.transform(X_test)
MNB_CV.fit(X_CV_train,y_train)
CV_y_predict=MNB_CV.predict(X_CV_test)
MNB_TV.fit(X_TV_train,y_train)
TV_y_predict=MNB_TV.predict(X_TV_test)

#rating
from sklearn.metrics import classification_report
print 'The behavior of CountVectorizer is:\r',classification_report(y_test,CV_y_predict,target_names=news.target_names)
print 'The behavior of TfidfVectorizer is:\r',classification_report(y_test,TV_y_predict,target_names=news.target_names)

