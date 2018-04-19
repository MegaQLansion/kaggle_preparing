# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:12:41 2018

@author: Lansion
"""

import tensorflow as tf
import numpy as np

greeting=tf.constant('Hello Tensorflow!')
sess=tf.Session()
result=sess.run(greeting)
print (result)
sess.close()

#calculating a linear function
#matrix1 is a 1x2 row vector, matrix2 is a 2x1 line vector
matrix1=tf.constant([[3.,3.]])
matrix2=tf.constant([[2.],[2.]])
#vector inner product
product=tf.matmul(matrix1,matrix2)
#add function
linear=tf.add(product,tf.constant(2.0))
with tf.Session() as sess:
    result=sess.run(linear)
    print (result)

#linear classifier
import pandas as pd
train=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/Breast-Cancer/breast-cancer-train.csv')
test=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/Breast-Cancer/breast-cancer-test.csv')
X_train=np.float32(train[['Clump Thickness','Cell Size']].T)
y_train=np.float32(train['Type'].T)
X_test=np.float32(test[['Clump Thickness','Cell Size']].T)
y_test=np.float32(test['Type'].T)

#defiine intercept and slope
b=tf.Variable(tf.zeros([1]))
k=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))

#define linear function
y=tf.matmul(k,X_train)+b

#define cost function
loss=tf.reduce_mean(tf.square(y-y_train))

#Gradient Descent, with step size of 0.01
optimizer=tf.train.GradientDescentOptimizer(0.01)
#least square
train=optimizer.minimize(loss)
#initialize variables
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
#loop
for step in range(0,1000):
    sess.run(train)
    if step %200==0:
        print (step,sess.run(k),sess.run(b))
#testing samples
test_negative=test.loc[test['Type']==0][['Clump Thickness','Cell Size']]
test_positive=test.loc[test['Type']==1][['Clump Thickness','Cell Size']]

#plot
import matplotlib.pyplot as plt
plt.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(test_positive['Clump Thickness'],test_positive['Cell Size'],marker='x',s=150,c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
lx=np.arange(0,12)

ly=(0.5-sess.run(b)-lx*sess.run(k)[0][0])/sess.run(k)[0][1]

plt.plot(lx,ly,color='green')
plt.show()