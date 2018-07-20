#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-25 16:08:32
# @Author  : ${guoshengkang} (${kangguosheng1@huokeyi.com})

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# else--> Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
import numpy as np
from numpy import *
from sklearn import datasets  
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat


def X2LSTMX(X=None):
	'''
	对于array X 转化为lstm的输入X 
	'''
	rows,cols=X.shape
	lstm_X=[]
	for row in range(rows):
		lstm_X.append([list(X[row])])
	return np.array(lstm_X, dtype=np.float32)
	# 以上代码等价于如下代码
	# X=X.reshape([-1,1,X.shape[1]])
	# X=X.astype(np.float32)


class LSTM(object):
	"""
	Parameters
	------------
	Attributes
	------------
	"""
	def __init__(self,HIDDEN_SIZE=50,NUM_LAYERS=5,BATCH_SIZE=32,TRAINING_STEPS=3000,
		learning_rate=0.1,optimizer ='Adagrad'):
		# 神经网络参数
		self.HIDDEN_SIZE = HIDDEN_SIZE  # LSTM隐藏节点个数
		self.NUM_LAYERS  = NUM_LAYERS   # LSTM层数
		self.BATCH_SIZE  = BATCH_SIZE   # batch大小
		# 数据参数
		self.TRAINING_STEPS = TRAINING_STEPS  # 训练轮数
		self.learning_rate = learning_rate # 学习率
		self.optimizer = optimizer
		self.regressor=None

	# LSTM结构单元
	def LstmCell(self):
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_SIZE)
		return lstm_cell

	def lstm_model(self,X, y):
		# 使用多层LSTM，不能用lstm_cell*NUM_LAYERS的方法，会导致LSTM的tensor名字都一样
		cell = tf.contrib.rnn.MultiRNNCell([self.LstmCell() for _ in range(self.NUM_LAYERS)])
		# 将多层LSTM结构连接成RNN网络并计算前向传播结果
		output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
		output = tf.reshape(output, [-1, self.HIDDEN_SIZE])
		# 通过无激活函数的全联接层计算线性回归，并将数据压缩成一维数组的结构
		predictions = tf.contrib.layers.fully_connected(output, 1, None)
		# 将predictions和labels调整为统一的shape
		y = tf.reshape(y, [-1])
		predictions = tf.reshape(predictions, [-1])
		# 计算损失值,使用平均平方误差
		loss = tf.losses.mean_squared_error(predictions, y)
		# 创建模型优化器并得到优化步骤
		train_op = tf.contrib.layers.optimize_loss(
			loss,
			tf.train.get_global_step(),
			optimizer=self.optimizer,
			learning_rate=self.learning_rate)
		return predictions, loss, train_op

	def fit(self,train_X=None,train_y=None):
		# 注意:train_X的size必须是[-1,1,col_num]
		train_X=train_X.reshape([-1,1,train_X.shape[1]])
		train_X=train_X.astype(np.float32)
		# 建立深层循环网络模型
		self.regressor = SKCompat(tf.contrib.learn.Estimator(model_fn=self.lstm_model))
		# 调用fit函数训练模型
		self.regressor.fit(train_X, train_y, batch_size=self.BATCH_SIZE, steps=self.TRAINING_STEPS)

	def predict(self,test_X):
		# 注意:train_X的size必须是[-1,1,col_num]
		test_X=test_X.reshape([-1,1,test_X.shape[1]])
		test_X=test_X.astype(np.float32)
		# 使用训练好的模型对测试集进行预测
		predicted = array([pred for pred in self.regressor.predict(test_X)])
		return predicted

if __name__ == "__main__":
    print('{:*^60}'.format('Input Data'))
    house_dataset = datasets.load_boston();    #加载波士顿房价数据集
    house_data = house_dataset.data;           #加载房屋属性参数
    house_price = house_dataset.target;        #加载房屋均价
    X_train,X_test,y_train,y_test=train_test_split(house_data,house_price,test_size=0.2,random_state=0)
    print('X_train:',X_train.shape,'y_train:',y_train.shape)
    print('X_test:',X_test.shape,'y_test:',y_test.shape)
    
    print('{:*^60}'.format('Call LSTM'))
    lstm = LSTM()
    lstm.fit(X_train, y_train)
    y_pre=lstm.predict(X_test)
    print('{:-^30}'.format('Predicted Value'))
    print(y_pre)

