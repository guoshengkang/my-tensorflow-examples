train samples: (800, 6212)
test samples: (200, 6212)

以下3个结果应该一致,前6位一致,后三位有误差(保留9位小数)
1.模型运行完后直接对测试数据的预测得分
code:my_tf.py
output:prediction_value.txt
2.保存参数(w1...w5,b1...b5)后,自己撰写函数对测试数据的预测得分
code:test_score.py
output:test_score.txt
3.加载保存的模型后直接对测试数据的预测得分
code:restore_my_tf.py
output:restore_prediction_value.txt

以下2个结果应该一致,前6位一致,后三位有误差(保留9位小数)
1.加载保存的模型后直接样本数据的预测得分
code:evaluation_by_model.py
output:uid_label_score_from_model.txt
2.保存参数(w1...w5,b1...b5)后,自己撰写函数对样本数据的预测得分
code:evaluation_by_parameters.py
output:uid_label_score_from_parameters.txt