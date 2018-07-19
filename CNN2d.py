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
from sklearn import datasets            #鸢尾花数据集被sklearn的datasets所包含，需要引用
from sklearn.cross_validation import train_test_split
if sys.version_info[0] == 3:
    from functools import reduce
else:
    pass
import numpy as np
import tensorflow as tf


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
        self.fnn_size=[10, 5, 1]

    def defineCNN(self,inputSize=None):
        """
        定义卷积神经网络的结构
        """
        img = tf.reshape(self.input, [-1, 1, inputSize, 1])
        # 定义卷积层1和池化层1，其中卷积层1里有4个feature map
        # convPool1的形状为[-1, 1, Xdim-3+1, 4] --> [-1, 1, (Xdim-3+1)/2, 4]
        convPool1 = self.defineConvPool(img, filterShape=[1, 3, 1, 4],
            poolSize=[1, 1, 2, 1])
        # 输出convPool1的维度信息
        # print(convPool1.get_shape().as_list())
        # 定义卷积层2和池化层2，其中卷积层2里有8个feature map
        # convPool2的形状为[-1, 4, 4, 40]
        convPool2 = self.defineConvPool(convPool1, filterShape=[1, 3, 4, 8],
            poolSize=[1, 1, 2, 1])
        # 输出convPool2的维度信息
        # print(convPool2.get_shape().as_list())
        # 将池化层2的输出变成行向量，后者将作为全连接层的输入
        pool_shape=convPool2.get_shape().as_list()
        convPool2 = tf.reshape(convPool2, [-1, pool_shape[1]*pool_shape[2]*pool_shape[3]])
        # 定义全连接层
        self.out = self.defineFullConnected(convPool2, size=self.fnn_size)

    def defineConvPool(self, inputLayer, filterShape, poolSize):
        """
        定义卷积层和池化层
        """
        weights = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1))
        # 将模型中的权重项记录下来，用于之后的惩罚项
        self.W.append(weights)
        biases = tf.Variable(tf.zeros(filterShape[-1]))
        # 定义卷积层
        _conv2d = tf.nn.conv2d(inputLayer, weights, strides=[1, 1, 1, 1], padding="VALID")
        convOut = tf.nn.relu(_conv2d + biases)
        # 输出convOut的维度信息
        # print(convOut.get_shape().as_list())
        # 定义池化层
        poolOut = tf.nn.max_pool(convOut, ksize=poolSize, strides=poolSize, padding="SAME")
        return poolOut

    def defineFullConnected(self, inputLayer, size):
        """
        定义全连接层的结构
        """
        prevSize = inputLayer.shape[1].value
        prevOut = inputLayer
        layer = 1
        # 定义隐藏层
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
        # 定义输出层
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

    def fit(self, X=None, Y=None, startLearningRate=0.1, miniBatchFraction=0.1, epoch=3, keepProb=0.8):
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
            prob = prob.reshape([max(prob.shape),]) # 转成nx1数组,不会改变输入参数的值
        return prob


if __name__ == "__main__":

    # 训练数据集 
    # X_train = np.load('X_train.npy')   
    # print("train samples:",X_train.shape)
    # y_train = np.load('y_train.npy') 
    # train_num = y_train.shape[0]
    # y_train = y_train.reshape([train_num,1]) # 转成nx1数组
    # print(y_train.shape)
    # # 测试数据集
    # X_test = np.load('X_test.npy')   
    # print("test samples:",X_test.shape)
    # y_test = np.load('y_test.npy') 
    # test_num = y_test.shape[0]
    # y_test = y_test.reshape([test_num,1]) # 转成nx1数组
    # print(y_test.shape)
    # ann = CNN()
    # ann.fit( X=X_train, Y=y_train)
    # y_pre=ann.predict(X=X_test)
    # print(y_pre)
    # print(y_test)

    house_dataset = datasets.load_boston();    #加载波士顿房价数据集
    house_data = house_dataset.data;           #加载房屋属性参数
    house_price = house_dataset.target;        #加载房屋均价
    X_train,X_test,y_train,y_test=train_test_split(house_data,house_price,test_size=0.2,random_state=0)
    y_train = y_train.reshape([len(y_train),1]) # 转成nx1数组
    y_test = y_test.reshape([len(y_test),1]) # 转成nx1数组
    ann = CNN()
    ann.fit( X=X_train, Y=y_train)
    y_pre=ann.predict(X=X_test)
    print(y_pre)
    # print(y_test)


