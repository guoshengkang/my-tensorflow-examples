#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os,sys,re
from pandas import Series,DataFrame
import datetime
reload(sys)
sys.setdefaultencoding('utf-8')
starttime = datetime.datetime.now()    

#######################################################
dictionary_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "dict.txt")
dictionary=[]
with open(dictionary_path, "r") as fin:
  for line in fin.readlines():
    line=unicode(line.strip(), "utf-8")
    dictionary.append(line)
print "There are %d keywords in the dictionary file!!!"%len(dictionary)

samples_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "sample_1000")
fin=open(samples_path)
lines=fin.readlines()
row_num=len(lines) #文件的行数
col_num=len(dictionary)
print "There are %d lines in the input file!!!"%row_num

df=DataFrame(np.zeros((row_num,col_num)),columns=dictionary)

for row,line in enumerate(lines): #row：0,1,2,3,...
  line=unicode(line.strip(),'utf-8')
  uid,label,keywords,province,tier,head,tail=line.split(unicode(',','utf-8'))
  keyword_list=keywords.split(unicode(';','utf-8'))
  df.ix[row,"label"]=label
  for keyword in keyword_list:
    if keyword in dictionary:
      df.ix[row,keyword]=1
  keyword_province='province_'+province
  keyword_tier='tier_'+tier
  keyword_head='head_'+head
  keyword_tail='tail_'+tail
  for keyword in [keyword_province,keyword_tier,keyword_head,keyword_tail]:
    if keyword in dictionary:
      df.ix[row,keyword]=1

fin.close()
df.fillna(0,inplace=True) #默认不为NAN,而是为0'province_'+province
print df.shape #输出表格的行列数

columns_path=os.path.join(os.path.split(os.path.realpath(__file__))[0], "column_names.txt")
columns_fout=open(columns_path,'w')
for column_name in df.columns: #将列名写到文件
    columns_fout.write(column_name+'\n')
columns_fout.close()

df_path=os.path.join(os.path.split(os.path.realpath(__file__))[0], "df_1000.csv")
df.to_csv(df_path,index=False) #将表格写到文件
#####################################################

endtime = datetime.datetime.now()
print (endtime - starttime),"time used!!!" #0:00:00.280797