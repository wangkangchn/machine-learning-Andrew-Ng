"""


				机器学习---支持向量机练习
		
		
"""

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from svmutil import *
import svmFunctions as sf

## =============== Part 1: 加载以及可视化数据 ================
#

print('加载以及可视化数据...\n')


load_data = sio.loadmat('ex6data1.mat')

X = load_data["X"]
y = load_data['y']
sf.plotData(X, y);


## ==================== Part 2: 训练线性核SVM ====================
# 使用svmutil 训练

print('训练线性核SVM...\n')

# 数据整理 第一列为标签, 其后为特征 
with open('train_dataset_1.txt', 'w') as f:
	for label, feature in zip(y, X):
		# 第一列加入标记
		msg = '{} '.format(int(label))
		for i in range(feature.shape[0]):
			# 后面加入特征
			msg += '{}:{} '.format(i+1, feature[i])
		msg += '\n'
		f.write(msg)
		
# 数据准备      
train_label, train_value = svm_read_problem("train.txt")         #训练数据集

# 训练数据
prob = svm_problem(train_label, train_value)
# 模型参数-c 1 C=1  -t 0线性内核
param = svm_parameter('-c 1 -t 0')
# 进行模型的训练 
model = svm_train(prob, param)

# 预测
p_labs, p_acc, p_vals = svm_predict(train_label, train_value, model)
# ~ print('p_labs: \n', p_labs)
# ~ print('p_acc: \n', p_acc)
# ~ print('p_vals: \n', p_vals)


## ========== Part 4: 使用高斯核训练数据集2 ==========
#

print('加载数据...\n')

# 加载数据集2的数据
load_data = sio.loadmat('ex6data2.mat')
X = load_data["X"]
y = load_data['y']

print('使用高斯核训练数据集2...\n');

train_fn = 'train_dataset_2.txt'

# 数据整理 第一列为标签, 其后为特征 
with open(train_fn, 'w') as f:
	for label, feature in zip(y, X):
		# 第一列加入标记
		msg = '{} '.format(int(label))
		for i in range(feature.shape[0]):
			# 后面加入特征
			msg += '{}:{} '.format(i+1, feature[i])
		msg += '\n'
		f.write(msg)
		
train_label, train_value = svm_read_problem(train_fn)
prob = svm_problem(train_label, train_value)

param = svm_parameter('-c 1 -t 2')
model = svm_train(prob, param)
svm_predict(train_label, train_value, model)


## ========== Part 5: 使用高斯核训练模型 ==========
#

print('加载数据...\n')

# 加载数据集2的数据
load_data = sio.loadmat('ex6data3.mat')
X = load_data["X"]
y = load_data['y']

Xval = load_data["Xval"]
yval = load_data['yval']

print('使用高斯核训练数据集3...\n');

train_fn = 'train_dataset_3.txt'

# 数据整理 第一列为标签, 其后为特征 
with open(train_fn, 'w') as f:
	for label, feature in zip(y, X):
		# 第一列加入标记
		msg = '{} '.format(int(label))
		for i in range(feature.shape[0]):
			# 后面加入特征
			msg += '{}:{} '.format(i+1, feature[i])
		msg += '\n'
		f.write(msg)

var_fn = 'var_dataset_3.txt'

# 整理验证数据 第一列为标签, 其后为特征 
with open(var_fn, 'w') as f:
	for label, feature in zip(y, X):
		# 第一列加入标记
		msg = '{} '.format(int(label))
		for i in range(feature.shape[0]):
			# 后面加入特征
			msg += '{}:{} '.format(i+1, feature[i])
		msg += '\n'
		f.write(msg)
		
train_label, train_value = svm_read_problem(train_fn)
val_label, val_value = svm_read_problem(var_fn)

C, g = sf.dataset3Params(train_value, train_label, val_value, val_label)

print('最佳: C: {}, g: {}'.format(C, g))
# Train the SVM
model= svm_train(train_label, train_value, '-c {} -t 2 -g {}'.format(C, g))



