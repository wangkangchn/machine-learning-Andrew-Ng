"""
				协同过滤算法

"""
import numpy as np
from matplotlib import pyplot as plt

class CollaborativeFiltering:
	""" 协同过滤算法 """
	
	def __init__ (self, params, Y, R, num_users, num_movies, num_features, ilambda=0):
		self.Y = Y
		self.R = R
		self.num_users = num_users
		self.num_movies = num_movies
		self.num_features = num_features
		self.ilambda = ilambda
		self.X, self.Theta = self.get_param(params)
		
	def get_param(self, params):
		""" 分离X, Theta """
		X = params[:self.num_movies * self.num_features].reshape(
				self.num_movies, self.num_features)
		Theta = params[self.num_movies * self.num_features:].reshape(
				self.num_users, self.num_features)
		return X, Theta
				
	def cofi_costfunc(params, Y, R, num_users, num_movies, num_features, ilambda=0):
		""" 代价函数 params为X和Theta组成的长向量"""
		X = params[:num_movies * num_features].reshape(num_movies, num_features)
		Theta = params[num_movies * num_features:].reshape(num_users, num_features)
		
		X_grad = np.zeros(X.shape)
		Theta_grad = np.zeros(Theta.shape)
		
		# 先进行计算所有值的预测误差, 再与R元素相乘提取所需要的元素
		J = (0.5 * np.sum(np.sum(np.power(R * (X @ Theta.T - Y), 2))) 
				+ ilambda / 2 * np.sum(np.sum(np.power(X, 2))) 
				+ ilambda / 2 * np.sum(np.sum(np.power(Theta, 2))))
		
		# 计算梯度
		X_grad = (R * (X @ Theta.T - Y)) @ Theta + ilambda * X			# N_m*N_u   N_u*n
		Theta_grad = (R * (X @ Theta.T - Y)).T @ X + ilambda * Theta	# N_u*N_m   N_m*n
		
		# 将梯度展开为一个长向量
		grad = np.vstack((X_grad.reshape(X_grad.size, 1), 
						Theta_grad.reshape(Theta_grad.size, 1)))
		
		return J, grad
	
	def gradient_descent(self, alpha, num_iters):
		""" 梯度下降 """
		J_history = np.zeros(num_iters)
		for iter in range(num_iters):		
			# 计算代价	
			J_history[iter] = (0.5 * np.sum(np.sum(np.power(self.R 
				* (self.X @ self.Theta.T - self.Y), 2))) 
				+ self.ilambda / 2 * np.sum(np.sum(np.power(self.X, 2))) 
				+ self.ilambda / 2 * np.sum(np.sum(np.power(self.Theta, 2))))
				
			# 计算梯度
			self.Theta -= alpha * ((self.R * (self.X @ self.Theta.T 
					- self.Y)).T @ self.X + self.ilambda * self.Theta)
			self.X -= alpha * ((self.R * (self.X @ self.Theta.T 
					- self.Y)) @ self.Theta + self.ilambda * self.X)
			print("Iteration   {} | Cost: {:.6e}".format(iter+1, J_history[iter]))
		#绘制迭代次数及对应的J值图像
		self.plot_data(np.arange(num_iters), J_history)

	def plot_data (self, x, y):
		""" 绘制代价图像 """
		plt.plot(x, y)
		plt.title('Cost')	
		plt.xlabel('times')
		plt.ylabel('J')
		
		
	def run(self, alpha=0.01, num_iters=50):
		""" 运行协同过滤算法, 学习特征以及参数 """
		# 开始交互式显示图片
		plt.ion()
		# 进行学习
		self.gradient_descent(alpha, num_iters)
	
		# 关闭交互式显示图片
		plt.ioff()
		plt.show()
