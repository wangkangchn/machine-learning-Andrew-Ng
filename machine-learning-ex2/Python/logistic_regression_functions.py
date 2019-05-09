"""
				逻辑回归相关函数

"""
import numpy as np

def map_feature(X1, X2):
	""" 进行多项式特征的映射 """	
	degree = 6;
	out = np.ones((X1.shape[0], 1))	#计算行数
	for i in range(1, degree+1):
	    for j in range(0, i+1):
	       out = np.hstack((out, np.power(X1, (i-j)) * np.power(X2, j)))	  
	return out

def feature_normalize (X):
	""" 
		进行特征值的均值归一化 
		
		以二维行向量形式返回mu, sigma
	"""
	mu = X.mean(0)             # 每列特征值均值
	sigma = X.std(0)           # 每列特征值标准差
	X_norm = (X - mu) / sigma
	return mu.reshape(1, mu.size), sigma.reshape(1, sigma.size), X_norm
