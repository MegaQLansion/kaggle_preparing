# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:03:30 2018

@author: Lansion
"""

from sklearn.datasets import fetch_20newsgroups
import numpy as np
news=fetch_20newsgroups(subset='all')

#split data sets
from sklearn.cross_validation import train_test_split
X=news.data[:3000]
y=news.target[:3000]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#SVM\TfidifVectorizer initialization
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

#connecting feature extraction with svm
clf=Pipeline([('vect',TfidfVectorizer(stop_words='english',analyzer='word')),('svc',SVC())])

#setting parameters 4x3=12 combinations
parameters={'svc__gamma':np.logspace(-2,1,4),'svc__C':np.logspace(-1,1,3)}

#import grid search
from sklearn.grid_search import GridSearchCV

#3 folds cross validation,Search with single thread
GS=GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3)

#search and print the best
GS.fit(X_train,y_train)
GS.best_params_,GS.best_score_
print 'The best parameters combination is',GS.best_params_,'with an accuracy of',GS.score(X_test,y_test)