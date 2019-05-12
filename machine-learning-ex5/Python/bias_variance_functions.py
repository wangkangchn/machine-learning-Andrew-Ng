""" 
		
		偏差方差评估所需函数
		
"""
import numpy as np
from matplotlib import pyplot as plt
from linear_regression import LinearRegression as LR

def learning_curve(X, y, Xval, yval, ilambda):
	""" 计算学习曲线所需的参数, 返回训练误差, 交叉验证误差"""
	
	# 随机选择训练样本以及验证样本, 重复50次取误差的平均值
	num_items = 50
	
	# 训练集大小
	m = X.shape[0]
	
	# 训练误差以及交叉验证误差
	error_train = np.zeros((m, 1))
	error_val   = np.zeros((m, 1))

	for i in range(m):	
		# 建立线性回归模型
		my_lr = LR(X[0:i+1, :], y[0:i+1])	
		my_lr.gradient_descent_reg(0.001, ilambda, 5000)
		theta = my_lr.theta
		error_train[i:], _ = my_lr.compute_cost_reg(theta, 0, X=X[:i+1, :], y=y[:i+1])
		error_val[i:], _ = my_lr.compute_cost_reg(theta, 0, X=Xval, y=yval)

	# ~ for i in range(m):	
		# ~ print("样本 {}".format(i+1))
		
		# ~ for t in range(num_items):	
			# ~ # 随机获取训练以及交叉验证样本
			# ~ rand_indices = np.arange(m)			# 随机获取样本的100行, 进行可视化
			# ~ np.random.shuffle(rand_indices)		# shuffle返回None 就地打乱
			# ~ sel_1 = rand_indices[:i+1]

			# ~ np.random.shuffle(rand_indices)
			# ~ sel_2 = rand_indices[:i+1]

			# ~ # 建立线性回归模型
			# ~ my_lr = LR(X[sel_1, :], y[sel_1])	
			
			# ~ # 使用不用的训练样本进行参数学习 -> 进行验证
			# ~ my_lr.gradient_descent_reg(0.001, ilambda, 5000)
			# ~ theta = my_lr.theta
			
			# ~ # 使用不同训练样本计算训练误差, lambda=0
			# ~ cost_train, _ = my_lr.compute_cost_reg(theta, 0, X=X[sel_1, :], y=y[sel_1])
			# ~ error_train[i:] += cost_train
			# ~ # 使用全部交叉验证样本计算交叉验证误差, lambda=0
			# ~ cost_val, _ = my_lr.compute_cost_reg(theta, 0, X=Xval[sel_2, :], y=yval[sel_2])
			# ~ error_val[i:] += cost_val
	
	# ~ # 计算误差平均值		
	# ~ error_train /= num_items
	# ~ error_val /= num_items
	
	return error_train, error_val


def poly_features(X, p):
	""" 进行多项式的映射 """
	X_poly = np.zeros((X.size, p))

	for i in range(1, p + 1):
		# 只使用整数返回的是向量, 进行切片才返回二维数组
		X_poly[:, i - 1:] = np.power(X, i)	

	return X_poly

def feature_normalize(X):
	""" 特征规范化 """	
	mu = X.mean(0)				# 每列特征值均值
	sigma = X.std(0)			# 每列特征值标准差
	X_norm = (X - mu) / sigma	
	return X_norm, mu, sigma

def plot_fit(min_x, max_x, mu, sigma, theta, p):
	""" 进行多项式数据的拟合 """
	x = np.arange(min_x - 15, max_x + 25, 0.05)
	x = x.reshape(x.size, 1)
	
	X_poly = poly_features(x, p)
	X_poly = (X_poly - mu) / sigma
	
	X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))     # 加入 1 列

	plt.plot(x, X_poly @ theta, '--', LineWidth=2)

def validation_curve(X, y, Xval, yval):
	""" 自动选择lambda """
	lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
	lambda_vec = lambda_vec.reshape(lambda_vec.size, 1)
	

	error_train = np.zeros((lambda_vec.shape[0], 1))
	error_val = np.zeros((lambda_vec.shape[0], 1))
	
	for i in range(lambda_vec.size):
		# 遍历lambda, 计算每一个的训练以及验证误差				
		ilambda = lambda_vec[i]
		
		#建立模型, 获得参数, 计算误差
		my_lr = LR(X, y)	
		my_lr.gradient_descent_reg(0.001, ilambda, 5000)
		theta = my_lr.theta
		error_train[i:], _ = my_lr.compute_cost_reg(theta, 0, X=X, y=y)
		error_val[i:], _ = my_lr.compute_cost_reg(theta, 0, X=Xval, y=yval)
			
	return  lambda_vec, error_train, error_val
