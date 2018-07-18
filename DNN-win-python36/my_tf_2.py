#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf  
from numpy.random import RandomState  
from functools import reduce
import numpy as np
import math
import datetime
import time
starttime = datetime.datetime.now()    
###################################
# train samples: (800, 6212)
# test samples: (200, 6212)
# 参数设置
x_dim = 6212 # 样本数据的维度
unit_num = 4 # 每个隐层包含的神经元数
batch_size = 100 # 定义每次训练数据batch的大小,防止内存溢出 
steps = 1000 # 设置神经网络的迭代次数 
top_percentage = 0.2 # 用于统计前%20得分的正负样本数
learning_rate = 0.02 # 学习率
regularizer_weight=0.001 # 正则在损失函数中权重
###################################

#定义输入和输出  
x = tf.placeholder(tf.float32,shape=(None,x_dim),name="x-input")  
y = tf.placeholder(tf.float32,shape=(None,1),name="y-input")  

# 方式二：神经网络循环定义
#######################################################################
ANN_size=[unit_num,unit_num,unit_num,unit_num,1]
prevSize=x_dim
prevOut=x
W=[]
B=[]
# 定义隐藏层
for currentSize in ANN_size[:-1]:
    weights=tf.Variable(tf.random_normal([prevSize,currentSize],stddev=1.0/np.sqrt(float(prevSize)),seed=1))
    W.append(weights)
    biases=tf.Variable(tf.zeros([1,currentSize]))
    B.append(biases)
    prevOut=tf.nn.tanh(tf.matmul(prevOut,weights) + biases)
    prevSize=currentSize
# 定义输出层
weights=tf.Variable(tf.random_normal([prevSize,ANN_size[-1]],stddev=1.0/np.sqrt(float(prevSize)),seed=1))
W.append(weights)
biases=tf.Variable(tf.zeros([1,ANN_size[-1]]))
B.append(biases)
prevOut=tf.nn.sigmoid(tf.matmul(prevOut,weights) + biases)
prediction=prevOut

# 训练数据集 
X_train = np.load('X_train.npy')   
print("train samples:",X_train.shape)
y_train = np.load('y_train.npy') 
train_num = y_train.shape[0]
top20_train = int(top_percentage*train_num)
y_train = y_train.reshape([train_num,1]) # 转成nx1数组
dataset_size=train_num # 训练样本数目

# 测试数据集
X_test = np.load('X_test.npy')   
print("test samples:",X_test.shape)
y_test = np.load('y_test.npy') 
num = y_test.shape[0]
print(y_test.shape)
top20_test = int(top_percentage*num)
y_test = y_test.reshape([num,1]) # 转成nx1数组

# 定义损失函数,这里只需要刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y - prediction))
_norm=map(lambda x:tf.nn.l2_loss(x),W)
regularization=reduce(lambda a,b:a+b,_norm)
loss=mse_loss+regularizer_weight*regularization

#定义反向传播算法的优化函数
# global_step=tf.Variable(0)
# decay_steps = int(math.ceil(train_num/batch_size)) #衰减速度
# decay_rate = 0.96 # 衰减系数
# learning_rate = tf.train.exponential_decay(0.1,global_step,decay_steps,decay_rate,staircase=True)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# 参数输出格式
fmt=['%.9f']
w1_fmt=unit_num*fmt
w2_fmt=unit_num*fmt
w3_fmt=unit_num*fmt
w4_fmt=unit_num*fmt
w5_fmt=fmt

saver=tf.train.Saver()

#创建会话运行TensorFlow程序  
with tf.Session() as sess:  
    #初始化变量  tf.initialize_all_variables()  
    init = tf.initialize_all_variables()  
    sess.run(init)  
    for i in range(steps):  
        #每次选取batch_size个样本进行训练  
        start = (i * batch_size) % dataset_size  
        end = min(start + batch_size,dataset_size)
        #通过选取样本训练神经网络并更新参数  
        X_batch=X_train[start:end];y_batch=y_train[start:end]
        #每迭代100次输出一次日志信息  
        if i % 100 == 0 :
            _,total_loss=sess.run([train_step,loss],feed_dict={x:X_batch,y:y_batch})
            print("time:%s,"%(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))),end='') # 打印当前时间
            # print "learning_rate:%0.6f,"%sess.run(learning_rate), # 打印学习率
            # 计算训练数据的损失之和  
            print("training_steps:%05d, total_loss:%0.6f,"%(i,total_loss),end='')
            # 对测试数据进行预测
            prediction_value = sess.run(prediction,feed_dict={x:X_test}) # mx1
            c= np.c_[y_test,prediction_value] # 将标签和得分合在一个数组中
            sorted_c=c[np.lexsort(-c.T)] # 按最后一列逆序排序
            print("[%0.6f,%0.6f],"%(sorted_c[top20_test-1,1],sorted_c[0,1]),end='')
            positive_num=sum(sorted_c[:top20_test,0]);negative_num=top20_test-positive_num
            print("positive_rate:%0.6f,"%(positive_num/top20_test),end='')
            prediction_label = np.array([[1 if p_value[0]>=0.5 else 0] for p_value in prediction_value]) # mx1
            yes_no=(prediction_label==y_test)
            print("prediction_accuracy:%0.6f on test data"%(yes_no.sum()/float(num)))
        else:
            sess.run(train_step,feed_dict={x:X_batch,y:y_batch}) 
    saver.save(sess,"./model/model.ckpt") #保存模型

    prediction_value = sess.run(prediction,feed_dict={x:X_test}) #预测得分
    np.savetxt('testdata_score_after_run.txt',prediction_value,fmt=fmt,delimiter=',') 
    
    #模型训练结束,输出和保存参数
    for i in range(len(ANN_size)):
        parameter_wi=W[i].eval(session=sess)
        parameter_bi=B[i].eval(session=sess)
        np.savetxt('w%d.txt'%(i+1),parameter_wi,fmt=ANN_size[i]*fmt,delimiter=',') 
        np.savetxt('b%d.txt'%(i+1),parameter_bi,fmt=ANN_size[i]*fmt,delimiter=',') 
   
writer=tf.summary.FileWriter("./graph",tf.get_default_graph())
writer.close()


endtime = datetime.datetime.now()
print((endtime - starttime),"time used!!!") #0:00:00.280797

print("Finished!!!")