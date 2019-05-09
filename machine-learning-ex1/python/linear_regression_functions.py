""" 
				机器学习: 线性回归算法相关函数
"""
import numpy as np

def feature_normalize (X):
	""" 进行特征值的均值归一化 """
	mu = np.mean(X, axis=0)		# 每列特征值均值
	sigma = np.std(X, axis=0)	# 每列特征值标准差
	Xnorm = (X - mu) / sigma	
	return mu, sigma, Xnorm

def map_feature (X1, X2, degree=6):
	""" 进行多项式特征的映射 """	
	out = np.ones((X1.shape[0], 1))	
	for i in range(1, degree+1):
	    for j in range(1, i+1):
	       out = np.hstack((out, np.power(X1, (i-j)) * np.power(X2, j)))	  
	return out
