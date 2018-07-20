#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-18 15:44:10
# @Author  : ${guoshengkang} (${kangguosheng1@huokeyi.com})

# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用tensorflow来实现CNN，
并在训练过程中使用防止过拟合的方案
"""
import os
import sys
from sklearn import datasets  
if sys.version_info[0] == 3:
    from functools import reduce
else:
    pass
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class CNN(object):

    def __init__(self, regularizer_weight=0.001):
        """
        创建一个卷积神经网络
        全连接中dropout的keepProb训练时在fit()传入,预测时在predict()中传入1.0
        """
        # 重置tensorflow的graph，确保神经网络可多次运行
        tf.reset_default_graph()
        tf.set_random_seed(1908)
        self.regularizer_weight = regularizer_weight
        self.W = []
        self.fnn_size=[8, 4, 1]

    def defineCNN(self,inputSize=None):
        """
        定义卷积神经网络的结构
        """
        inputLayer = tf.reshape(self.input, [-1, inputSize, 1])
        # (1) 定义卷积层1和池化层1，其中卷积层1里有4个feature map
        convPool1 = self.defineConvPool(inputLayer, channels=8,conv_size=4,conv_strides=1,
        pool_size=2,pool_strides=2)
        print('convPool1:',convPool1.get_shape().as_list())
        # (2) 定义卷积层2和池化层2，其中卷积层2里有8个feature map
        convPool2 = self.defineConvPool(convPool1, channels=12,conv_size=4,conv_strides=1,
        pool_size=2,pool_strides=2)
        print('convPool2:',convPool2.get_shape().as_list())
        # (3) 将池化层2的输出变成行向量，后者将作为全连接层的输入
        pool_shape=convPool2.get_shape().as_list()
        convPool2 = tf.reshape(convPool2, [-1, pool_shape[1]*pool_shape[2]])
        print('transformed convPool2:',convPool2.get_shape().as_list())
        # (4) 定义全连接层
        self.out = self.defineFullConnected(convPool2, size=self.fnn_size)

    def defineConvPool(self, inputLayer, channels=8,conv_size=4,conv_strides=1,
        pool_size=2,pool_strides=2):
        """
        定义卷积层和池化层
        inputLayer:输入tensor
        channels:卷积层数
        conv_size:卷积窗口大小
        conv_strides:卷积步长
        pool_size:池化窗口大小
        pool_strides:池化步长
        """
        # (1) Convolution
        # inputs:输入一个tensor。
        # filters:integer：输出空间的维度。
        # kernel_size:An integer or tuple/list of a single integer，1维卷积窗口的大小。
        # strides: An integer or tuple/list of a single integer，卷积步长。
        # padding: One of `"valid"` or `"same"`
        conv = tf.layers.conv1d(inputs=inputLayer, filters=8, kernel_size=4, strides=1, padding="same", activation=tf.nn.relu)
        # print('conv:',conv.get_shape().as_list())
        # (2) Pooling
        pool = tf.layers.max_pooling1d(inputs=conv, pool_size=2, strides=2, padding="same")
        # print('pool:',pool.get_shape().as_list())
        return pool

    def defineFullConnected(self, inputLayer, size):
        """
        定义全连接层的结构
        """
        prevSize = inputLayer.shape[1].value
        prevOut = inputLayer
        layer = 1
        # (1) 定义隐藏层
        for currentSize in size[:-1]:
            weights = tf.Variable(
                tf.truncated_normal([prevSize, currentSize], stddev=1.0 / np.sqrt(float(prevSize))),
                name="fc%s_weights" % layer)
            # 将模型中的权重项记录下来，用于之后的惩罚项
            self.W.append(weights)
            biases = tf.Variable(
                tf.zeros([currentSize]),
                name="fc%s_biases" % layer)
            layer += 1
            # 定义这一层神经元的输出
            neuralOut = tf.nn.relu(tf.matmul(prevOut, weights) + biases)
            # 对隐藏层里的神经元使用dropout
            prevOut = tf.nn.dropout(neuralOut, self.keepProb)
            prevSize = currentSize
        # (2) 定义输出层
        weights = tf.Variable(tf.truncated_normal(
            [prevSize, size[-1]], stddev=1.0 / np.sqrt(float(prevSize))),
            name="output_weights")
        # 将模型中的权重项记录下来，用于之后的惩罚项
        self.W.append(weights)
        biases = tf.Variable(tf.zeros([size[-1]]), name="output_biases")
        out = tf.matmul(prevOut, weights) + biases
        return out

    def defineLoss(self):
        """
        定义神经网络的损失函数
        """
        # 定义单点损失，self.label是训练数据里的标签变量
        loss = tf.reduce_mean(tf.square(self.label - self.out))
        # L2惩罚项
        _norm = map(lambda x: tf.nn.l2_loss(x), self.W)
        regularization = reduce(lambda a, b: a + b, _norm)
        # 定义整体损失
        self.loss = tf.reduce_mean(loss + self.regularizer_weight * regularization,name="loss")
        # 记录训练的细节
        tf.summary.scalar("loss", self.loss)
        return self

    def SGD(self, X, Y, startLearningRate, miniBatchFraction, epoch, keepProb):
        """
        使用随机梯度下降法训练模型
        """
        trainStep = tf.Variable(0)
        learningRate = tf.train.exponential_decay(startLearningRate, trainStep,
            1000, 0.96, staircase=True)
        method = tf.train.GradientDescentOptimizer(learningRate)
        optimizer= method.minimize(self.loss, global_step=trainStep)
        batchSize = int(X.shape[0] * miniBatchFraction)
        batchNum = int(np.ceil(1 / miniBatchFraction)) 
        sess = tf.Session()
        self.sess = sess
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        while (step < epoch):
            for i in range(batchNum):
                batchX = X[i * batchSize: (i + 1) * batchSize]
                batchY = Y[i * batchSize: (i + 1) * batchSize]
                sess.run([optimizer], feed_dict={self.input: batchX, self.label: batchY, self.keepProb: keepProb})
            step += 1
        return self

    def fit(self, X=None, Y=None, startLearningRate=0.1, miniBatchFraction=0.1, epoch=3, keepProb=0.7):
        """
        训练模型
        """
        if len(Y.shape)==1:
            Y = Y.reshape([len(Y),1]) # 转成nx1数组,不会改变输入参数的值
        self.inputSize = X.shape[1]
        self.input = tf.placeholder(tf.float32, shape=[None, self.inputSize], name="X")
        self.label = tf.placeholder(tf.float32, shape=[None, 1], name="Y")
        self.keepProb = tf.placeholder(tf.float32)
        self.defineCNN(inputSize=self.inputSize)
        self.defineLoss()
        self.SGD(X, Y, startLearningRate, miniBatchFraction, epoch, keepProb)

    def predict(self, X):
        """
        使用神经网络对未知数据进行预测
        """
        sess = self.sess
        prob = sess.run(self.out, feed_dict={self.input: X, self.keepProb: 1.0})
        if len(prob.shape)==2:
            prob = prob.reshape([max(prob.shape),]) # 转成向量,不会改变输入参数的值
        return prob


if __name__ == "__main__":
    # # 训练数据集 
    # X_train = np.load('X_train.npy')   
    # print("train samples:",X_train.shape)
    # y_train = np.load('y_train.npy') 
    # print(y_train.shape)
    # # 测试数据集
    # X_test = np.load('X_test.npy')   
    # print("test samples:",X_test.shape)
    # y_test = np.load('y_test.npy') 
    # print(y_test.shape)
    # # 调用CNN
    # ann = CNN()
    # ann.fit( X=X_train, Y=y_train)
    # y_pre=ann.predict(X=X_test)
    # print(y_pre)


    house_dataset = datasets.load_boston();    #加载波士顿房价数据集
    house_data = house_dataset.data;           #加载房屋属性参数
    house_price = house_dataset.target;        #加载房屋均价
    X_train,X_test,y_train,y_test=train_test_split(house_data,house_price,test_size=0.2,random_state=0)
    print('X_train:',X_train.shape)
    print(y_train.shape)
    ann = CNN()
    ann.fit( X=X_train, Y=y_train)
    y_pre=ann.predict(X=X_test)
    print(y_pre)