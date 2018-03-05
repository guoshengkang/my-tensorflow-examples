train samples: (800, 6212)
test samples: (200, 6212)

★★★【计算测试数据的得分】★★★
以下3个结果应该一致,前6位一致,后三位有误差(保留9位小数)
1.模型运行完后直接对测试数据的预测得分
code:my_tf.py
output:testdata_score_after_run.txt
2.保存参数(w1...w5,b1...b5)后,自己撰写函数对测试数据的预测得分
code:testdata_score_by_parameters.py
output:testdata_score_by_parameters.txt
3.加载保存的模型后直接对测试数据的预测得分
code:testdata_score_by_model.py
output:testdata_score_by_model.txt

★★★【计算样本数据的得分】★★★
以下2个结果应该一致,前6位一致,后三位有误差(保留9位小数)
1.加载保存的模型后直接样本数据的预测得分
code:sampledata_score_by_model.py
output:sampledata_score_by_model.txt
2.保存参数(w1...w5,b1...b5)后,自己撰写函数对样本数据的预测得分
code:sampledata_score_by_parameters.py
output:sampledata_score_by_parameters.txt