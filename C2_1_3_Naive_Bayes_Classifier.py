# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:55:28 2018

@author: Lansion
"""

#downloading initial data sets
from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups(subset='all')
X=news.data
y=news.target


#split data sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#feature extraction to text(equals feature scaling to numbers)
from sklearn.feature_extraction.text import CountVectorizer
CV=CountVectorizer()
X_train=CV.fit_transform(X_train)
X_test=CV.transform(X_test)

#initialize training environment
from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()

#training with NBC
MNB.fit(X_train,y_train)
MNB_y_predict=MNB.predict(X_test)
print 'The accuracy of Naive Bayes is',MNB.score(X_test,y_test)

#F1score
from sklearn.metrics import classification_report
print 'The score of Naive Bayes is:\r',classification_report(y_test,MNB_y_predict,target_names=news.target_names)