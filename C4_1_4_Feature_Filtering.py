
import pandas as pd
titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
y=titanic['survived']
X=titanic.drop(['row.names','survived','name'],axis=1)
#complete age using average value
X['age'].fillna(X['age'].mean(),inplace=True)
X.fillna('UNKNOWN',inplace=True)
#split data sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)
#feature extraction
from sklearn.feature_extraction import DictVectorizer
DV=DictVectorizer()
X_train=DV.fit_transform(X_train.to_dict(orient='record'))
X_test=DV.transform(X_test.to_dict(orient='recored'))

#feature filtering
from sklearn import feature_selection
FS=feature_selection.SelectPercentile(feature_selection.chi2,percentile=20)
X_train_FS=FS.fit_transform(X_train,y_train)
X_test_FS=FS.transform(X_test)
print 'The original dimension of features is',len(DV.feature_names_)
#initialization
from sklearn.tree import DecisionTreeClassifier
DTC=DecisionTreeClassifier(criterion='entropy')
DTC_FS=DecisionTreeClassifier(criterion='entropy')
#training with decision tree
DTC.fit(X_train,y_train)
DTC_y_predict=DTC.predict(X_test)
#training with decision tree using feature selecting
DTC_FS.fit(X_train_FS,y_train)
DTC_FS_y_predict=DTC_FS.predict(X_test_FS)
#rating
from sklearn.metrics import classification_report
print 'The accuracy of decision tree with original feature is',DTC.score(X_test,y_test)
print 'The accuracy of decision tree with Feature Selecting of 20% is',DTC_FS.score(X_test_FS,y_test)
#looking for the optimal percentile of features
from sklearn.cross_validation import cross_val_score
import numpy as np
percentiles=range(1,100,1)
results=[]
for i in percentiles:
    FS=feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
    X_train_FS=FS.fit_transform(X_train,y_train)
    DTC_FS.fit(X_train_FS,y_train)
    scores=cross_val_score(DTC_FS,X_train_FS,y_train,cv=5)
    results=np.append(results,scores.mean())
opt=np.where(results==results.max())[0]
print 'The optimal percentile of features is',opt[0],'%'
#rating using optimal percentile
FS=feature_selection.SelectPercentile(feature_selection.chi2,percentile=opt[0])
X_train_FS=FS.fit_transform(X_train,y_train)
DTC_FS.fit(X_train_FS,y_train)
X_test_FS=FS.transform(X_test)
print 'The accuracy of decision tree with feature selecting of ',opt[0],'% is',DTC_FS.score(X_test_FS,y_test)
#print 'The accuracy of decision tree with Feature Selecting of',opt[0],'% is',results[opt[0]]
#plotting
import pylab as pl
pl.plot(percentiles,results)
pl.xlabel('percentiles of ')
pl.ylabel('accuracy')
pl.show()
