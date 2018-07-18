#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-22 16:40:20
# @Author  : ${guoshengkang} (${kangguosheng1@huokeyi.com})

import os
import numpy as np
from numpy import *
from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf

# tf中包装了各种各样的机器学习模型
learn=tf.contrib.learn

def my_model(features,target):
	target=tf.one_hot(target,3,1,0)
	logits,loss=learn.models.logistic_regression(features,target)
	train_op=tf.contrib.layers.optimize_loss(
		loss,
		tf.contrib.framework.get_global_step(),
		optimizer='Adagrad',
		learning_rate=0.1)
	return tf.arg_max(logits,1),loss,train_op

iris=datasets.load_iris()
X=iris.data   #array (150,4)
y=iris.target #array (150,)

x_train,x_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)
# print(x_train.shape) #(120, 4)
classifier=learn.Estimator(model_fn=my_model)
classifier.fit(x_train,y_train,steps=100)

y_predicted=classifier.predict(x_test)
y_predicted=array([e for e in y_predicted])
print(y_predicted)

score=metrics.accuracy_score(y_test,y_predicted)
print('Accuracy:%.2f%%'%(score*100))