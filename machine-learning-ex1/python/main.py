import numpy as np
from matplotlib import pyplot as plt
from threading import Thread
import linear_regression_functions as lrf
from linear_regression import LinearRegression as lr

def error (name, value, y, m):
	e = (value - y).T @ (value - y) / (2*m)
	print('{}预测误差为: 	{:e} '.format(name, float(e)))
	
def batch(X, y, p, batch_num_iters, ilambda):
	name = '批量梯度下降'
	print('{}'.format(name))
	# 参数设置
	alpha = 0.01
	
	# 进行特征值的均值归一化
	mu, sigma, X_norm = lrf.feature_normalize (X)
	# 添加偏置单元
	X_norm = np.hstack((np.ones((m, 1)), X_norm))
	my_linear = lr(X_norm, y)
	
	# ~ print('\n\n{}学习中(ilambda = {})...'.format(name, ilambda))
	my_linear.batch_gradient_descent(alpha, batch_num_iters, ilambda)
	
	# ~ theta = ''
	# ~ for val in my_linear.theta:
		# ~ theta += str(val)[str(val).find('[')+1:-1]
		# ~ theta += '\n'
	# ~ print('θ: \n{}'.format(theta))
	
	# ~ print('{}学习结束...'.format(name))
	# 对预测值进行归一化, 并且添加偏置单元
	p_norm = (p - mu) / sigma
	p_norm = np.hstack((np.ones((p_norm.shape[0], 1)), p_norm))
	value = p_norm @ my_linear.theta
	# 误差率
	error (name, value, y, m)

def eqn (X, y, p, ilambda):
	name = '正规方程'
	print('{}'.format(name))
	# 创建模型		
	my_linear = lr(X, y)
	
	# ~ print('\n\n{}学习中(ilambda = {})...'.format(name, ilambda))
	my_linear.normal_Eqn(ilambda)
	
	# ~ theta = ''
	# ~ for val in my_linear.theta:
		# ~ theta += str(val)[str(val).find('[')+1:-1]
		# ~ theta += '\n'
	# ~ print('θ: \n{}'.format(theta))
	
	# ~ print('{}学习结束...'.format(name))
	p_eqn = np.hstack((np.ones((p.shape[0], 1)), p))
	value = p_eqn @ my_linear.theta
	# 误差率
	error (name, value, y, m)

def stochastic (X, y, p, batch_num_iters, ilambda):
	name = '随机梯度下降'
	print('{}'.format(name))
	# 参数设置
	alpha = 0.01
	
	# 进行特征值的均值归一化
	mu, sigma, X_norm = lrf.feature_normalize (X)
	# 添加偏置单元
	X_norm = np.hstack((np.ones((m, 1)), X_norm))
	my_linear = lr(X_norm, y)
	
	# ~ print('\n\n{}学习中(ilambda = {})...'.format(name, ilambda))
	my_linear.stochastic_gradient_descent(alpha, stoch_num_iters, ilambda)
	
	# ~ theta = ''
	# ~ for val in my_linear.theta:
		# ~ theta += str(val)[str(val).find('[')+1:-1]
		# ~ theta += '\n'
	# ~ print('θ: \n{}'.format(theta))
	# ~ print('{}学习结束...'.format(name))
	
	# 对预测值进行归一化, 并且添加偏置单元
	p_norm = (p - mu) / sigma
	p_norm = np.hstack((np.ones((p_norm.shape[0], 1)), p_norm))
	value = p_norm @ my_linear.theta
	
	# 误差率
	error (name, value, y, m)

def mini (X, y, p, batch_num_iters, ilambda):		
	name = 'mini_batch梯度下降'
	print('{}'.format(name))
	# 参数设置
	alpha = 0.01
	
	b = 2
	# 进行特征值的均值归一化
	mu, sigma, X_norm = lrf.feature_normalize (X)
	# 添加偏置单元
	X_norm = np.hstack((np.ones((m, 1)), X_norm))
	my_linear = lr(X_norm, y)
	
	# ~ print('\n\n{}学习中(ilambda = {})...'.format(name, ilambda))
	my_linear.mini_batch_gradient_descent(b, alpha, mini_num_iters, ilambda)
	
	# ~ theta = ''
	# ~ for val in my_linear.theta:
		# ~ theta += str(val)[str(val).find('[')+1:-1]
		# ~ theta += '\n'
	# ~ print('θ: \n{}'.format(theta))
	# ~ print('{}学习结束...'.format(name))
	
	# 对预测值进行归一化, 并且添加偏置单元
	p_norm = (p - mu) / sigma
	p_norm = np.hstack((np.ones((p_norm.shape[0], 1)), p_norm))
	value = p_norm @ my_linear.theta
	# 误差率
	error (name, value, y, m)

# 加载数据
data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=int) #delimiter

# 训练集的大小
m = data.shape[0]	
X = data[:, 0:2]
y = data[:, -1].reshape((m,1))

# 预测数据
p = data[:, 0:2]	
		
ilambda = 0
batch_num_iters = 1000
stoch_num_iters = 5000
mini_num_iters = 5000

# 建立线程
threads = []
functions = [eqn, batch, stochastic, mini]
for i in range(len(functions)):
	if i == 0:
		thread = Thread(target=functions[i], args=(X, y, p, ilambda))
		threads.append(thread) 
	else:
		thread = Thread(target=functions[i], args=(X, y, p, batch_num_iters, ilambda))
		threads.append(thread) 
	
# 开启线程, 比较不同的梯度下降算法
for i in range(len(functions)):
	threads[i].start()



