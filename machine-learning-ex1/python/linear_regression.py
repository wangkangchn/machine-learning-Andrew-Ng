""" 

				机器学习: 线性回归算法
				
"""
import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:
	"""
				机器学习: 线性回归(单, 多变量)
	
		参数:
			m 		 - 		训练集大小
			X 		 -		训练集
			y 		 - 		标签	
			theta 	 -　	训练参数 
			
			alpha	 -		学习率
			num_iters-		迭代次数
			
			mu		 -		每列特征的均值
			sigma	 -		每列特征的标准差
		
		方法：
			
			compute_cost		  		- 	计算代价值
			
			batch_gradient_descent		-	批量梯度下降	
			stochastic_gradient_descent	-	随机梯度下降
			mini_batch_gradient_descent	-   Mini Batch梯度下降
			normal_Eqn					-	正规方程求解
			
			plot_cost					-   绘制代价函数图像
	"""
	
	def __init__ (self, X, y):
		# 训练集大小
		self.m = y.shape[0]				
		self.X = X							
		self.y = y
		# 在0-1内随机初始化参数
		self.theta = np.random.rand(self.X.shape[1], 1)	

	def compute_cost (self, theta, ilambda):
		""" 代价函数 """
		J = 1 / (2 * self.m) * ((self.X @ theta - self.y).T @ (self.X @ theta - self.y) + 
				ilambda * np.sum(np.power(theta[:, 1:], 2)))		#theta 从1开始计算
		return J
		
	def batch_gradient_descent (self, alpha, num_iters, ilambda=0):
		""" 批量梯度下降算法 """			
		#保存每次迭代的J值	
		# ~ J_history = np.zeros(num_iters)		
		
		for iter in range(num_iters):
			theta0_grad = 1 / self.m * (self.X[:, 0].T @ (self.X @ 
					self.theta - self.y)) 
			thetaj_grad = (1 / self.m * (self.X[:, 1:].T @ (self.X @ 
					self.theta - self.y)) 
					+ ilambda / self.m * np.sum(self.theta[1:]))
			
			#保持同时更新
			self.theta[0, :] -= alpha * theta0_grad
			self.theta[1:, :] -= alpha * thetaj_grad
			
			# ~ J_history[iter] = self.compute_cost(self.theta, ilambda)
		#绘制迭代次数及对应的J值图像
		# ~ self.plot_cost(np.arange(num_iters), J_history)
	
	def stochastic_gradient_descent (self, alpha, num_iters, ilambda=0):
		"""
		随机梯度下降算法(用于大数据集)
			
			每遍历一个样本, 便对参数θ进行更新一次, 在数据集很大的情况下,
		遍历一次数据集, 足以收敛至全局最小附近. 一般1~10次的整个数据集的
		迭代是可以接受的.
			调试: 每遍历1000个样本, 便绘制出前1000个样本的误差均值
			
		"""				
		# 开启交互显示图片
		# ~ plt.ion()
		# ~ plt.title('Cost')	
		# ~ plt.xlabel('times')
		# ~ plt.ylabel('J')

		cost = np.zeros(self.m)
		# 总的循环次数一般为1~10, 当数据量很大时, 遍历一次数据集即可
		for iter in range(num_iters):
			# 遍历样本进行梯度下降
			
			for i in range(self.m):	
				
				# 对参数进行更新前计算代价值 0.5*(h(x) - y)
				# ~ cost[i] = 0.5 * ((self.X[i, :] @ self.theta - self.y[i]) ** 2 + 
						# ~ ilambda * np.sum(np.power(self.theta[:, 1:], 2)))		
				
				# 每遍历一个样本都要进行参数的更新	
				theta0_grad = 1 / self.m * (self.X[i, 0] * (self.X[i, :] @ 
						self.theta - self.y[i])) 
				thetaj_grad = (1 / self.m * np.sum(self.X[i, 1:] * (self.X[i, :] @ 
						self.theta - self.y[i])) 
						+ ilambda / self.m * np.sum(self.theta[1:]))
				# 保持参数的同步更新
				self.theta[0, :] -= alpha * theta0_grad
				self.theta[1:, :] -= alpha * thetaj_grad
				
			# ~ plt.plot(iter + 1, np.mean(cost), 'ro')
		
		# 关闭交互显示
		# ~ plt.ioff()
		# ~ plt.show()
	
	def mini_batch_gradient_descent (self, b, alpha, num_iters, ilambda=0):
		"""
		mini_batch梯度下降算法(用于大数据集)
			
			每遍历 b(2~100) 个样本, 便对参数θ进行更新一次, 选择合适的参数b
		以及好的向量化实现, 速度会比随机梯度下降更加的快.
			调试: 每遍历1000个样本, 便绘制出前1000个样本的误差均值
			
		"""			
		# 总的循环次数一般为1~10, 当数据量很大时, 遍历一次数据集即可
		for iter in range(num_iters):
			# 遍历样本进行梯度下降
			
			for i in range(0, self.m, b):									
				# 每遍历b个样本进行更新	
				theta0_grad = 1 / self.m * (self.X[i:i+b, 0].T @ (self.X[i:i+b, :] @ 
						self.theta - self.y[i:i+b])) 
				thetaj_grad = (1 / self.m * np.sum(self.X[i:i+b, 1:].T @ (self.X[i:i+b, :] @ 
						self.theta - self.y[i:i+b])) 
						+ ilambda / self.m * np.sum(self.theta[1:]))
				
				#保持同时更新
				self.theta[0, :] -= alpha * theta0_grad
				self.theta[1:, :] -= alpha * thetaj_grad
		
		
	def normal_Eqn (self, ilambda):
		""" 正规方程算法 """		
		self.X = np.hstack((np.ones((self.m, 1)), self.X))	
		
		# 正则项, 比变量个数+1(偏置单元)
		reg = np.eye(self.X.shape[1])		
		reg[0, 0] = 0		# θo 不进行正则化

		self.theta =(np.linalg.inv(self.X.T @ self.X + 
				ilambda * reg) @ self.X.T @ self.y)	
		
	def plot_cost (self, x, y):
		""" 绘制代价图像 """
		plt.plot(x, y)
		plt.title('Cost')	
		plt.xlabel('times')
		plt.ylabel('J')
		plt.show()
	

		
		
		
		
		
		
		
