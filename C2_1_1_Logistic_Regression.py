#dealing with raw data
column_names=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chormatin','Normal Nucleoli','Mitoses','Class']
import pandas as pd
data=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
data=data.replace(to_replace='?',value=np.nan)
data=data.dropna(how='any')

#split data sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)


#feature scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

#initialize training environment
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
lr=LogisticRegression()
SGDC=SGDClassifier()


#training with logistic regression
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)
print 'The accuracy of logistic regression is',lr.score(X_test,y_test)

#training with SGDClassifier
SGDC.fit(X_train,y_train)
SGDC_y_predict=SGDC.predict(X_test)
print 'The accuracy of SGDClassifier is',SGDC.score(X_test,y_test)

#precision\recall\f1-score of lr and SGDC
from sklearn.metrics import classification_report
print 'The score of logistic regression is:\r',classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant'])
print 'The score of SGD is:\r',classification_report(y_test,SGDC_y_predict,target_names=['Benign','Malignant'])