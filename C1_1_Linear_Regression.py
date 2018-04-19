import pandas as pd
df_train=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/Breast-Cancer/breast-cancer-train.csv')
df_test=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/Breast-Cancer/breast-cancer-test.csv')
df_test_positive=df_test.loc[df_test['Type']==1]
df_test_negative=df_test.loc[df_test['Type']==0]
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
print lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type'])
a=lr.coef_[0,:]
print a[0:]