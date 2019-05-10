"""
				机器学习: 逻辑回归算法
"""
import numpy as np
import pylab
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LogisticRegression:
	"""
				机器学习: 逻辑回归(单, 多变量)

		参数:
			m           -      训练集大小
			X           -      训练集
			y           -      标记
			theta       -　    训练参数
			alpha       -      学习率
			ilambda     -       正则参数
			num_iters   -      迭代次数
			mu          -      每列特征的均值
			sigma       -      每列特征的标准差

		方法：
			feature_normalize    -   特征均值归一化
			init_parameterdata       -   初始化梯度下降中使用的参数 X, theta, J
			sigmoid             -   假设函数

			compute_cost         -   代价函数
			compute_cost      -   正则代价函数

			gradient_descent     -   梯度下降算法
			gradient_descentReg  -   正则梯度下降算法

			prediction        -   梯度下降算法预测
			accuracy            -   模型精度

			plot_data            -   绘制代价图像
	"""

	def __init__ (self, X, y):
		#为便于矩阵运算, 加入特征X0恒等于1
		self.m = y.shape[0]                 #训练集数目
		self.X = X                          #训练数据
		self.y = y                          #标记
		self.theta = np.zeros((self.X.shape[1], 1))

	def sigmoid (self, z):
		""" 假设函数 """
		return 1 / (1 + np.exp(-z))

	def compute_cost (self, theta, ilambda=0):
		""" 代价函数, 测试用 """
		J = (- 1 / self.m * (self.y.T @ np.log(self.sigmoid(self.X @ theta))
				+ (1 - self.y).T @ np.log(1 - self.sigmoid(self.X @ theta)))
				+ ilambda / (2 * self.m) * np.sum(np.power(theta[1:], 2), axis=0))

		# 计算梯度
		grad = np.zeros(theta.shape)
		grad[0] = 1 / self.m * self.X[:, 0:1].T @ (self.sigmoid(self.X @ theta) - self.y)
		grad[1:] = (1 / self.m * self.X[:, 1:].T @ (self.sigmoid(self.X @ theta) - self.y)
				+ ilambda / self.m * theta[1:])
		return J, grad

	def gradient_descent (self, alpha, ilambda, num_iters):
		""" 梯度下降算法 """
		# 保存每次迭代的代价值
		# ~ J_history = np.zeros(num_iters)

		#迭代
		for iter in range(num_iters):
			theta_0 = (self.theta[0]
					- alpha / self.m * (self.X[:, 0:1].T @ (self.sigmoid(self.X @ self.theta) - self.y)))
			theta_j = (self.theta[1:] * (1 - alpha  * ilambda / self.m)
					- alpha / self.m * (self.X[:, 1:].T @ (self.sigmoid(self.X @ self.theta) - self.y)))

			#保持θ同时更新
			self.theta[0] = theta_0
			self.theta[1:] = theta_j
			
			# ~ J_history[iter], _ = self.compute_cost(self.theta, ilambda)
			
			# 保留最后一次的代价
			if iter == num_iters-1:   
				self.J_end, _ = self.compute_cost(self.theta, ilambda)
		#绘制迭代次数及对应的J值图像
		# ~ self.plot_data(np.arange(num_iters), J_history)
		
		
	def prediction(self, X):
		""" 梯度下降算法预测 """
		return self.sigmoid(X @ self.theta)

	def accuracy (self):
		""" 返回模型精确度 """
		accu = self.X @ self.theta
		accu[accu >= 0] = 1
		accu[accu < 0] = 0
		p = accu[accu == self.y]
		return p.shape[0] / self.m

	def plot_data (self, x, y):
		""" 绘制代价图像 """
		plt.plot(x, y, c='g')
		plt.title('Cost')
		plt.xlabel('times')
		plt.ylabel('J(θ)')
		plt.show()
		
	def map_feature(self, X1, X2):
		""" 进行多项式特征的映射 """
		degree = 6;
		out = np.ones((X1.shape[0], 1)) #计算行数
		for i in range(1, degree+1):
			for j in range(0, i+1):
			   out = np.hstack((out, np.power(X1, (i-j)) * np.power(X2, j)))
		return out

	def plot_3D(self, X,Y, z,):
			#绘制3D图
			fig = plt.figure()
			ax = Axes3D(fig)
			ax.plot_surface(X,Y, z, cmap=plt.cm.hot)  #cmap=plt.cm.hot
			ax.contour(X,Y, z,  colors='red')     #X,Y数据空间, 第三个参数为值   offset=-60,
			ax.set_title('Decision Boundary of 3D')
			ax.set_xlabel('Value 1',color='r') #设置x坐标
			ax.set_ylabel('Value 2',color='r')
			ax.set_zlabel('Z label',color='r')

	def plot_sample(self, data):
		""" 绘制样本 """
		#原始数据
		X = data[:, 0:2]
		pos = (data[:, 2] == 1)
		neg = (data[:, 2] == 0)

		#样本数据
		fig = plt.figure()
		plt.plot(X[pos][:, 0:1], X[pos][:, 1:2], 'g+')
		plt.plot(X[neg][:, 0:1], X[neg][:, 1:2], 'ro')
		plt.xlabel('value1')
		plt.ylabel('value2')
		plt.legend(['Admitted', 'Not admitted'])
		plt.title('Decision Boundary')

	def plot_decision_boundary(self, data):
		""" 绘制决策边界 """
		self.plot_sample(data)

		#线性
		if self.X.shape[1] <= 3:
			# Only need 2 points to define a line, so choose two endpoints
			plot_x = np.array([np.min(self.X[:,2])-2,  np.max(self.X[:,2])+2])

			# Calculate the decision boundary line
			plot_y = (- 1 / self.theta[2]) * (self.theta[1] * plot_x + self.theta[0])

			#Plot, and adjust axes for better viewing
			plt.plot(plot_x, plot_y)

			# Legend, specific for the exercise
			plt.legend('Decision Boundary')
			plt.axis([30, 100, 30, 100])

		#多项式
		else:
			# Here is the grid range
			u = np.linspace(-1, 1.5, 50).reshape(50, 1)
			v = np.linspace(-1, 1.5, 50).reshape(50, 1)
			# 生成网格数据
			X, Y = np.meshgrid(u,v)

			# Evaluate z = theta*x over the grid
			z = np.zeros((u.shape[0], v.shape[0]))
			for i in range(u.shape[0]):
				for j in range(v.shape[0]):
					#对每一个网格都进行多项式的计算, numpy下标一个数字仅仅是向量, 需要进行切片才能成为二维向量!!!!
					z[i,j] = self.map_feature(u[i:i+1], v[j:j+1]) @ self.theta

			# X,Y数据空间, 第三个参数为值 [显示的数据]
			contour = plt.contour(X, Y, z, [0], colors='black')
			# ~ self.plot_3D(X, Y, z)

		plt.show()


