# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 17:35:55 2018

@author: Lansion
"""

import pandas as pd
digits_train=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)
#split data sets
X_digits=digits_train[np.arange(64)]
y_digits=digits_train[64]
X_test=digits_test[np.arange(64)]
y_test=digits_test[64]
#initialization
from sklearn.decomposition import PCA
estimator=PCA(n_components=2)
#train
X_pca=estimator.fit_transform(X_digits)
#visualization
from matplotlib import pyplot as plt
def plot_pca_scatter():
    colors=['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']
    for i in xrange(len(colors)):
        px=X_pca[:,0][y_digits.as_matrix()==i]
        py=X_pca[:,1][y_digits.as_matrix()==i]
        plt.scatter(px,py,c=colors[i])
    
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
    
plot_pca_scatter()

#separate data sets
X_train=digits_train[np.arange(64)]
y_train=digits_train[64]
X_test=digits_test[np.arange(64)]
y_test=digits_test[64]
estimator=PCA(n_components=20)
PCA_X_train=estimator.fit_transform(X_train)
PCA_X_test=estimator.transform(X_test)
#initialize
from sklearn.svm import LinearSVC
SVC=LinearSVC()
PCA_SVC=LinearSVC()
#train with original data and data after pca
SVC.fit(X_train,y_train)
SVC_y_predict=SVC.predict(X_test)
PCA_SVC.fit(PCA_X_train,y_train)
PCA_SVC_y_predict=PCA_SVC.predict(PCA_X_test)
#rating with score
from sklearn.metrics import classification_report
print 'The accuracy of original data trained with Linear Support Vector Classifier is',SVC.score(X_test,y_test)
print 'The accuracy of data after PCA trained with Linear Support Vector Classifier is',PCA_SVC.score(PCA_X_test,y_test)



