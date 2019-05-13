"""
			支持向量机所需要的函数

"""

import numpy as np
from matplotlib import pyplot as plt
from svmutil import *

def plotData(X, y):
	
	# 找到正负样本点下标, where返回行列元组
	pos = np.where(y == 1)[0]	
	neg = np.where(y == 0)[0]

    # 数据显示
	plt.plot(X[pos, 0], X[pos, 1], 'r+', LineWidth=1, MarkerSize=7)
	plt.plot(X[neg, 0], X[neg, 1], 'go', MarkerSize=7)

# ~ def visualizeBoundary(X, y, model, varargin):	
	# ~ # 原始数据集
	# ~ plotData(X, y)
	
	# ~ # 绘制绘图表格
	# ~ x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)	# 行
	# ~ x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
	
	# ~ # 生成网格数据
	# ~ X1, X2 = np.meshgrid(x1plot, x2plot)	# 行 列
	# ~ # 保存所有的数据
	# ~ vals = np.zeros(X1.size)		# 每个网格点的数据都进行保存
	# ~ for i in range(X1.shape[1]):
		# ~ # 将数据进行拼接起来
		# ~ this_X = np.hstack((X1[:, i], X2[:, i]))
		# ~ # 预测
		# ~ _, _, p_vals = svm_predict(0, this_X, model)
	
	# ~ plt.contour(X1, X2, p_vals, [0.5], colors='black')
	# ~ plt.show()
	
def visualizeBoundaryLinear(X, y, model):

	# 生成绘制曲线的数据
	xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
	xp = xp.reshape(xp.size, 1)

	msg ='' 
	with open('test.txt', 'w') as f:
		for i in range(xp.shape[0]):
			# 后面加入特征
			msg += '{} 1:{} 2:{} '.format(0, float(xp[i]), float(xp[i]))
			# ~ print(msg)
			msg += '\n'
			f.write(msg)
			msg = '' # 清零
			
	test_label, test_value = svm_read_problem("test.txt")         #训练数据集	
	# 进行数据的预测
	_, _, p_vals = svm_predict(test_label, test_value, model)
	plotData(X, y)
	plt.plot(xp, p_vals, '-b')


def dataset3Params(X, y, Xval, yval):
	C = 1;
	sigma = 0.3;
	
	# 可供选择C, sigma的值
	value = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
	# 保存每一组C, sigma的误差
	error = np.zeros((value.size, value.size))
	
	# 选择C
	for c in range(value.size): 		
		# 选择sigma
		for s in range(value.size):		
			print('C: {}, sigma: {}'.format(value[c], value[s]))
			# 使用C sigma训练模型
			model= svm_train(y, X, '-c {} -t 2 -g {}'.format(value[c], 
						1 / (value[s] * value[s] * 2))) 
			# 进行预测
			_, e, _ = svm_predict(yval, Xval, model)
			error[c, s] = e[1]
			# 计算误差	
			print('error: {}\n\n'.format(error[c, s]))
	
	
	# 提取最小的误差
	dummy = np.min(np.min(error))
	# 查找微小误差的位置
	row, col = np.where(error==dummy)
	C = value[row]
	sigma = value[col]

	if isinstance(C, np.ndarray):
		C = C[0]
	if isinstance(sigma, np.ndarray):
		sigma = sigma[0]	
	print('min C: {}, min sigma: {}, min error = {}\n'.format(C, sigma, dummy))
	
	g = float(1 / (sigma * sigma * 2))
	
	return C, g
