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

★★★【tensorflow计算图的可视化】★★★
C:\mygithub\my-tensorflow-example>tensorboard --logdir=C:\mygithub\my-tensorflow-example\graph
2018-07-17 10:30:02.278426: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
W0717 10:30:02.303193 Reloader tf_logging.py:121] Found more than one graph event per run, or there was a metagraph containing a graph_def, as well as one or more graph events.  Overwriting the graph with the newest event.
W0717 10:30:02.304195 Reloader tf_logging.py:121] Found more than one metagraph event per run. Overwriting the metagraph with the newest event.
TensorBoard 1.8.0 at http://DESKTOP-UB4RBVB:6006 (Press CTRL+C to quit)
在浏览器输入:http://DESKTOP-UB4RBVB:6006 即可看到可视化的计算图
注:命令行相当于服务器的运行,若按CTRL+C则退出可视化服务

★★★【神经网络的定义】★★★
以下2种定义方式是等价的,模型参数的输出是一致的。
逐层定义神经网络的输出结果：
time:2018-07-18 09:59:56,training_steps:00000, total_loss:0.266210,[0.504263,0.618324],positive_rate:0.400000,prediction_accuracy:0.520000 on test data
time:2018-07-18 09:59:57,training_steps:00100, total_loss:0.199363,[0.464213,0.639435],positive_rate:0.575000,prediction_accuracy:0.610000 on test data
time:2018-07-18 09:59:57,training_steps:00200, total_loss:0.158241,[0.489225,0.718247],positive_rate:0.625000,prediction_accuracy:0.620000 on test data
time:2018-07-18 09:59:58,training_steps:00300, total_loss:0.134466,[0.504784,0.785377],positive_rate:0.700000,prediction_accuracy:0.640000 on test data
time:2018-07-18 09:59:58,training_steps:00400, total_loss:0.109549,[0.509013,0.837498],positive_rate:0.675000,prediction_accuracy:0.630000 on test data
time:2018-07-18 09:59:59,training_steps:00500, total_loss:0.096278,[0.538054,0.869158],positive_rate:0.675000,prediction_accuracy:0.630000 on test data
time:2018-07-18 09:59:59,training_steps:00600, total_loss:0.070854,[0.534508,0.888035],positive_rate:0.675000,prediction_accuracy:0.625000 on test data
time:2018-07-18 10:00:00,training_steps:00700, total_loss:0.071500,[0.584945,0.902113],positive_rate:0.675000,prediction_accuracy:0.605000 on test data
time:2018-07-18 10:00:00,training_steps:00800, total_loss:0.050650,[0.603390,0.911626],positive_rate:0.675000,prediction_accuracy:0.610000 on test data
time:2018-07-18 10:00:01,training_steps:00900, total_loss:0.055368,[0.624296,0.920436],positive_rate:0.675000,prediction_accuracy:0.610000 on test data
0:00:09.332801 time used!!!
Finished!!!
循环定义神经网络的输出结果：
time:2018-07-18 10:29:41,training_steps:00000, total_loss:0.266210,[0.504263,0.618324],positive_rate:0.400000,prediction_accuracy:0.520000 on test data
time:2018-07-18 10:29:42,training_steps:00100, total_loss:0.199363,[0.464213,0.639435],positive_rate:0.575000,prediction_accuracy:0.610000 on test data
time:2018-07-18 10:29:43,training_steps:00200, total_loss:0.158241,[0.489225,0.718247],positive_rate:0.625000,prediction_accuracy:0.620000 on test data
time:2018-07-18 10:29:44,training_steps:00300, total_loss:0.134466,[0.504784,0.785377],positive_rate:0.700000,prediction_accuracy:0.640000 on test data
time:2018-07-18 10:29:44,training_steps:00400, total_loss:0.109549,[0.509013,0.837498],positive_rate:0.675000,prediction_accuracy:0.630000 on test data
time:2018-07-18 10:29:45,training_steps:00500, total_loss:0.096278,[0.538054,0.869158],positive_rate:0.675000,prediction_accuracy:0.630000 on test data
time:2018-07-18 10:29:46,training_steps:00600, total_loss:0.070854,[0.534508,0.888035],positive_rate:0.675000,prediction_accuracy:0.625000 on test data
time:2018-07-18 10:29:46,training_steps:00700, total_loss:0.071500,[0.584945,0.902113],positive_rate:0.675000,prediction_accuracy:0.605000 on test data
time:2018-07-18 10:29:47,training_steps:00800, total_loss:0.050650,[0.603390,0.911626],positive_rate:0.675000,prediction_accuracy:0.610000 on test data
time:2018-07-18 10:29:47,training_steps:00900, total_loss:0.055368,[0.624296,0.920436],positive_rate:0.675000,prediction_accuracy:0.610000 on test data
0:00:07.507951 time used!!!
Finished!!!




