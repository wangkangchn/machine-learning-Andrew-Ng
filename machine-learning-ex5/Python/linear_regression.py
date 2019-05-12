""" 
				机器学习: 线性回归算法
"""
import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:
	"""
				机器学习: 线性回归(单, 多变量)
	
		参数:
			m 		 :  	训练集大小
			X 		 : 		训练集(需要包含偏置)
			y 		 : 		确定值
			theta 	 :　	参数 
			alpha	 :		学习率
			num_iters:		迭代次数
			mu		 :		每列特征的均值
			sigma	 :		每列特征的标准差
		
		方法：
			featureNormalize  	-	特征均值归一化
			mapFeature			-   进行特征值多项式映射
			
			computeCost		  	- 	计算代价值
			computeCostReg		- 	计算正则代价值
			
			gradientDescent		-	梯度下降算法	
			gradientDescentReg	-	正则梯度下降算法
			
			normalEqn			-	正规方程算法
			normalEqnReg		-	正则正规方程算法
			
			predictionGD		-	梯度下降算法预测
			predictionEqn		-	正规方程算法预测
			
			plotData			-   绘制代价函数图像
	"""
	
	def __init__ (self, X, y):
										#为便于矩阵运算, 加入特征X0恒等于1
		self.m = y.shape[0]				#训练集数目	
		self.X = X							
		self.y = y
		self.theta = np.zeros((self.X.shape[1], 1))	#与特征个数一致	
	
	def featureNormalize (self, X):
		""" 进行特征值的均值归一化 """
		self.mu = X.mean(0)				#每列特征值均值
		self.sigma = X.std(0)			#每列特征值标准差
		return (X - self.mu) / self.sigma	
	
	def mapFeature(self, X1, X2):
		""" 进行多项式特征的映射 """	
		degree = 6;
		out = np.ones((X1.shape[0], 1))	#计算行数
		for i in range(1, degree+1):
		    for j in range(1, i+1):
		       out = np.hstack((out, np.power(X1, (i-j)) * np.power(X2, j)))	  
		return out
		
	def computeCost (self, theta):
		""" 代价函数 """
		J = 1 / (2 * self.m) * ((self.X @ theta - self.y).T @ (self.X @ theta - self.y))
		return J
		
	def computeCostReg (self, theta, ilambda, X=None, y=None,):
		""" 计算代价与梯度, 供测试使用 """
		if X is not None and y is not None:
			# 传入数据, 使用传入的数据进行计算
			self.X = X
			self.y = y
		
		grad = np.zeros(theta.shape)
		
		J = (1 / (2 * self.m) * (self.X @ theta - self.y).T @ (self.X @ theta - self.y) + 
				ilambda / (2 * self.m) * np.sum(np.power(theta[1:], 2)))		#theta 从1开始计算
		
		# theta_0 单独计算
		grad[0] = 1 / self.m * self.X[:, 0].T @ (self.X @ theta - self.y)
		grad[1:] = (1 / self.m * self.X[:, 1:].T @ (self.X @ theta - self.y)
				+ ilambda / self.m * theta[1:])
				
		return J, grad
		
	def gradientDescent (self, alpha, num_iters, normalize=False):
		""" 梯度下降算法 """
		self.alpha = alpha				#学习率
		self.num_iters = num_iters
		
		if normalize:
			self.X = self.featureNormalize(self.X)	
		self.X = np.hstack((np.ones((self.m, 1)), self.X))
					
		J_history = np.zeros(self.num_iters)						#保存每次迭代的J值
		
		for iter in range(self.num_iters):
			self.theta = self.theta - self.alpha / self.m * (self.X.T @ (self.X @ self.theta - self.y))
			J_history[iter] = self.computeCost(self.theta)
		
		#绘制迭代次数及对应的J值图像
		self.plotData(np.arange(self.num_iters), J_history)
		
	def gradientDescentReg (self, alpha, ilambda, num_iters, normalize=False):
		""" 梯度下降算法 """
		if normalize:
			self.X = self.featureNormalize(self.X)		
					
		J_history = np.zeros(num_iters)						#保存每次迭代的J值
		
		for iter in range(num_iters):
			
			J_history[iter], grad = self.computeCostReg(self.theta, ilambda)
			
			self.theta[0] -= alpha * grad[0] 
			self.theta[1:] -= alpha * grad[1:]
		
		#绘制迭代次数及对应的J值图像
		# ~ self.plotData(np.arange(num_iters), J_history)
	
	def normalEqn (self):
		""" 正规方程算法 """		
		self.X = np.hstack((np.ones((self.m, 1)), self.X))	
		#np.linalg.inv(A)计算逆矩阵 numpy.linalg模块包含线性代数的函数
		self.theta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y	
	
	def normalEqnReg (self, ilambda):
		""" 正则化正规方程算法 """		
		self.X = np.hstack((np.ones((self.m, 1)), self.X))	
		
		reg = np.eye(self.X.shape[1])		#正则项, 比变量个数+1
		reg[0, 0] = 0

		self.theta = np.linalg.inv(self.X.T @ self.X + ilambda*reg) @ self.X.T @ self.y	
	
	def predictionGD(self, X):
		""" 梯度下降算法预测 """
		#先归一再计算
		X_n = (X - self.mu) / self.sigma
		X_p = np.hstack((np.ones((X_n.shape[0], 1)), X_n))
		return X_p @ self.theta 	
		
	def predictionEqn(self, X):
		""" 正规方程算法预测 """
		X_p = np.hstack((np.ones((X.shape[0], 1)), X))
		return X_p @ self.theta 	
		
	def plotData (self, x, y):
		""" 绘制代价图像 """
		plt.plot(x, y)
		plt.title('Cost')	
		plt.xlabel('times')
		plt.ylabel('J')
		plt.show()
	

		
		
		
		
		
		
		
