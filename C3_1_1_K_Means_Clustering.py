# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:37:39 2018

@author: Lansion
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#
digits_train=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)
#split data sets
X_train=digits_train[np.arange(64)]
y_train=digits_train[64]
X_test=digits_test[np.arange(64)]
y_test=digits_test[64]

#initialization
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=10)
#training
kmeans.fit(X_train)
kmeans_y_predict=kmeans.predict(X_test)
#rating with API
from sklearn import metrics
print 'The API score of kmeans is:',metrics.adjusted_rand_score(y_test,kmeans_y_predict)
#rating with Silhouette Coefficient
#plot
from sklearn.metrics import silhouette_score

clusters=[2,3,4,5,6,7,8]
sc_scores=[]
max_scores=0
for t in clusters:
    kmeans_model=KMeans(n_clusters=t).fit(X)
    sc_scores=silhouette_score(X,kmeans_model.labels_,metric='euclidean')
    print 'Cluster=',t,'has a score of',sc_scores
    if sc_scores>max_scores:
        max_scores=sc_scores
        idx=t
print 'The best cluster number is',idx,'with SC=',max_scores
    