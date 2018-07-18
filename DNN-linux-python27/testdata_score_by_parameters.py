#!/usr/bin/python
# -*- coding: utf-8 -*-
# import tensorflow as tf  
from numpy.random import RandomState  
import numpy as np
import math
import datetime
import time,os
from numpy import *
starttime = datetime.datetime.now()    
###################################
# 函数定义
def fun_tanh(x):
    y=(1.0-np.exp(-2.0*x))/(1.0+np.exp(-2.0*x))
    return y
def fun_sigmoid(x):
    y=1.0/(1.0+np.exp(-x))
    return y
def deep_learning_score(wb,x):
    # wb=[(w1,b1),(w2,b2),(w3,b3),(w4,b4)]
    x=mat(x)
    length=len(wb)
    for (w,b) in wb[:length-1]:
        x=fun_tanh(x*w+b)
    (w,b)=wb[-1]
    y=fun_sigmoid(x*w+b)
    return y[0,0] #返回一个数,而不是矩阵

w1=np.loadtxt('w1.txt',delimiter=',')
w2=np.loadtxt('w2.txt',delimiter=',')
w3=np.loadtxt('w3.txt',delimiter=',')
w4=np.loadtxt('w4.txt',delimiter=',')
w5=np.loadtxt('w5.txt',delimiter=',')
w5=w5.reshape(-1,1) #(128L,)-->(128L,1)
# print w1.shape,w2.shape,w3.shape,w4.shape
# (6416L, 128L) (128L, 128L) (128L, 128L) (128L, 1L)
b1=np.loadtxt('b1.txt',delimiter=',')
b2=np.loadtxt('b2.txt',delimiter=',')
b3=np.loadtxt('b3.txt',delimiter=',')
b4=np.loadtxt('b4.txt',delimiter=',') #<type 'numpy.ndarray'>
b5=np.loadtxt('b5.txt',delimiter=',') 
wb=[(w1,b1),(w2,b2),(w3,b3),(w4,b4),(w5,b5)]

# 参数输出格式
fmt=['%.9f']

fout_path=os.path.join(os.path.split(os.path.realpath(__file__))[0], "testdata_score_by_parameters.txt")
fout=open(fout_path,'w')
# 测试数据集
x = np.load('X_test.npy') 
(row_num,col_num)=x.shape
for index in range(row_num):
    y=deep_learning_score(wb,x[index,:])
    line="%.9f"%y
    fout.write(line+'\n')
########################################
endtime = datetime.datetime.now()
print (endtime - starttime),"time used!!!" #0:00:00.280797
print "Finished!!!"
