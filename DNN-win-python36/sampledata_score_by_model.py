#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf  
from numpy.random import RandomState  
import numpy as np
import math,os
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
###################################

#定义输入和输出  
x = tf.placeholder(tf.float32,shape=(None,x_dim),name="x-input")  
y = tf.placeholder(tf.float32,shape=(None,1),name="y-input")  

#定义神经网络的参数
#第一个隐层
w1 = tf.Variable(tf.random_normal([x_dim,unit_num],stddev=1,seed=1)) #6235行,10列
b1 = tf.Variable(tf.zeros([1,unit_num])) # 1,10
z1 = tf.matmul(x, w1)+b1 # 
a1 = tf.nn.tanh(z1) #使用tanh函数作为激活函数
#第二个隐层
w2 = tf.Variable(tf.random_normal([unit_num,unit_num])) #10行10列
b2 = tf.Variable(tf.zeros([1,unit_num])) # 1,10
z2 = tf.matmul(a1, w2)+b2 # 
a2 = tf.nn.tanh(z2) #使用tanh函数作为激活函数
#第三个隐层
w3 = tf.Variable(tf.random_normal([unit_num,unit_num])) #10行10列
b3 = tf.Variable(tf.zeros([1,unit_num])) # 1,10
z3 = tf.matmul(a2, w3)+b3 # 
a3 = tf.nn.tanh(z3) #使用tanh函数作为激活函数
#第四个隐层
w4 = tf.Variable(tf.random_normal([unit_num,unit_num])) #10行10列
b4 = tf.Variable(tf.zeros([1,unit_num])) # 1,10
z4 = tf.matmul(a3, w4)+b4 # 
a4 = tf.nn.tanh(z4) #使用tanh函数作为激活函数
#定义神经网络输出层
w5 = tf.Variable(tf.random_normal([unit_num,1]))
b5 = tf.Variable(tf.zeros([1,1]))
z5 = tf.matmul(a4,w5) + b5
prediction = tf.nn.sigmoid(z5)

# 定义损失函数,这里只需要刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y - prediction))
# 这个函数第一个参数是'losses'是集合的名字,第二个参数是要加入这个集合的内容
tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(0.001)(w1))
tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(0.001)(w2))
tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(0.001)(w3))
tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(0.001)(w4))
# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses',mse_loss)
# 将集合中的元素加起来,得到最终的损失函数
loss=tf.add_n(tf.get_collection('losses'))

#定义反向传播算法的优化函数
# global_step=tf.Variable(0)
# decay_steps = int(math.ceil(train_num/batch_size)) #衰减速度
# decay_rate = 0.96 # 衰减系数
# learning_rate = tf.train.exponential_decay(0.1,global_step,decay_steps,decay_rate,staircase=True)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# 参数输出格式
fmt=['%.9f']

saver=tf.train.Saver()
#########################################
dictionary_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "dict-label.txt")
dictionary=[]
with open(dictionary_path, "r") as fin:
  for line in fin.readlines():
    line=unicode(line.strip(), "utf-8")
    dictionary.append(line)
print "There are %d keywords in the dictionary file!!!"%len(dictionary)

samples_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "sample_20")
fin=open(samples_path)
lines=fin.readlines()
row_num=len(lines) #文件的行数
col_num=len(dictionary)
print "There are %d lines in the input file!!!"%row_num

fout_path=os.path.join(os.path.split(os.path.realpath(__file__))[0], "sampledata_score_by_model.txt")
fout=open(fout_path,'w')
#########################################
#创建会话运行TensorFlow程序  
with tf.Session() as sess:  
    saver.restore(sess,"/home/kang/Desktop/my_tf/model/model.ckpt")
    for row,line in enumerate(lines): # row：0,1,2,3,...
      tmp_arr=np.zeros((1,col_num)) # 初始化(1L, 6416L) tmp_arr.dtype=float64
      line=unicode(line.strip(),'utf-8')
      uid,label,keywords,province,tier,head,tail=line.split(unicode(',','utf-8'))
      keyword_list=keywords.split(unicode(';','utf-8'))
      for keyword in keyword_list:
        if keyword in dictionary:
          index=dictionary.index(keyword)
          tmp_arr[0,index]=1.0
      keyword_province='province_'+province
      keyword_tier='tier_'+tier
      keyword_head='head_'+head
      keyword_tail='tail_'+tail
      for keyword in [keyword_province,keyword_tier,keyword_head,keyword_tail]:
        if keyword in dictionary:
          index=dictionary.index(keyword)
          tmp_arr[0,index]=1.0
      input_x=tmp_arr
      y = sess.run(prediction,feed_dict={x:input_x})
      new_line="%s,%s,%.9f"%(uid,label,y)
      fout.write(new_line+'\n')
    fin.close()
    fout.close()

endtime = datetime.datetime.now()
print (endtime - starttime),"time used!!!" #0:00:00.280797
print "Finished!!!"