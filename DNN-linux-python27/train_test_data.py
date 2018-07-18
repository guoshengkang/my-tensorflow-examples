#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
from numpy import *

data=pd.read_csv('df_1000.csv')
print data.shape
row,col = data.shape
y=data.iloc[:,0].values
X=data.iloc[:,1:col].values
print X.shape,y.shape

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) # MemoryError
print "X_train:",X_train.shape
print "X_test:",X_test.shape

# X_test,X_train,y_test,y_train=(X[:50000,:],X[50000:,:],y[:50000],y[50000:])
# print "X_train:",X_train.shape
# print "X_test:",X_test.shape

np.save('X_train.npy',X_train)
np.save('X_test.npy',X_test)
np.save('y_train.npy',y_train)
np.save('y_test.npy',y_test)
