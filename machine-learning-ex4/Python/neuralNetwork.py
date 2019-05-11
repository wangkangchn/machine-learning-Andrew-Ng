"""

				机器学习 之 神经网络(3层 400 25 10)
	
	参数说明:
		
		nn_params			-	列向量, 包含所有需要的参数(将每个层间参数按行展开为列向量)	
		Theta1				-	输入层与隐藏层层间参数(权重)
		Theta2				-	隐藏层与输出层层间参数(权重)
		
		input_layer_size	-	输入层单元个数(特征数量)
		hidden_layer_size	-	隐藏层单元个数
		num_labels			-	标记数量(输出层单元的个数)
		
		X					-	特征集
		y					-	标记集
		
		ilambda				-	正则参数
		alpha				-	学习率
		num_items			-	迭代学习次数
		
	
	方法功能说明:
	
		get_theta			-	从参数长列向量中提取层间参数(权重)
		recode_y			-	将标记重新编码为对应分类的二进制向量
		
		sigmoid				-	激活函数
		sigmoid_gradient	-	激活函数的导数
		
		cost_function		-	用于梯度检测(依据传入的参数theta进行单次的代价和梯度的计算)
		feed_forward		-	前向传播, 返回h_x以及cost
		back_propagation	-	反向传播, 返回各参数梯度
		
		gradient_descent	-	梯度下降寻找使代价最小的参数theta
		mini_batch_gradient_descent
"""

import numpy as np

class NeuralNetwork:
	""" 三层神经网络 """

	def __init__ (self, nn_params, input_layer_size, hidden_layer_size,
			num_labels, X, y, ilambda=0):
		self.input_layer_size = input_layer_size
		self.hidden_layer_size = hidden_layer_size
		self.num_labels = num_labels

		self.X = X
		self.m = self.X.shape[0]
		self.y = self.recode_y(y)	

		self.Theta1, self.Theta2 = self.get_theta(nn_params)
		self.ilambda = ilambda

	def get_theta(self, nn_params):
		""" 获取层间参数Theta """
		Theta1 = nn_params[0 : self.hidden_layer_size * (self.input_layer_size+1), :].reshape(
				self.hidden_layer_size, self.input_layer_size + 1)
		Theta2 = nn_params[self.hidden_layer_size * (self.input_layer_size+1):, :].reshape(
				self.num_labels, self.hidden_layer_size + 1)
		return Theta1, Theta2

	def recode_y(self, y):
		""" 对标记y重新编码, 变为num_labels维向量的形式 """
		y_recode = np.zeros((self.m, self.num_labels))
		# ~ print('np.where(y==0): {}'.format(np.where(y==0)[0].shape))			# where返回行列下标组成的元组!!!		
		for i in range(self.num_labels):
			y_recode[np.where(y==i)[0], i] = 1			
		return y_recode

	def sigmoid(self, t):
		return 1 / (1 + np.exp(-t))

	def sigmoid_gradient (self, t):
		return self.sigmoid(t) * (1 - self.sigmoid(t))
		
	def cost_function (self, nn_params=None):
		""" 
			为了进行梯度检测, 计算传播一次的代价值与梯度有参数传来, 使用
		新参数计算, 没有参数使用旧参数计算
		"""
		if nn_params is not None:
			self.Theta1, self.Theta2 = self.get_theta(nn_params)
			
		Theta1_grad = np.zeros(self.Theta1.shape)
		Theta2_grad = np.zeros(self.Theta2.shape)
		
		# Step 1 前向传播计算h(x)
		a_1 = np.hstack((np.ones((self.m, 1)), self.X))             #5000*401
		z_2 = a_1 @ self.Theta1.T                                   #5000*25
		a_2 = np.hstack((np.ones((self.m, 1)), self.sigmoid(z_2)))  #5000*26
		z_3 = a_2 @ self.Theta2.T                                   #5000*10
		h_x = self.sigmoid(z_3)                                     #5000*10

		# Step 2 计算代价
		J = (- 1 / self.m * np.sum(np.sum(np.eye(self.m) * (self.y @ np.log(h_x.T) 
				+ (1 - self.y) @ np.log(1 - h_x.T)))) 
				+ self.ilambda / (2 * self.m) * (np.sum(np.sum(np.power(self.Theta1[:, 1:], 2))) 
				+ np.sum(np.sum(np.power(self.Theta2[:, 1:], 2)))))

		# Step 3 反向传播 之 计算输出层的偏导数
		delta_3 = h_x - self.y          #5000*10 Theta1 25*401 Theta2 10*26

		# Step 4 反向传播 之 计算对隐藏层输入z的偏导数
		delta_2 = (delta_3 @ self.Theta2)[:, 1:] * self.sigmoid_gradient(z_2)    #5000*25

		# Step 5 反向传播 之 计算参数的偏导数
		Theta2_grad += (delta_3.T @ a_2) / self.m   #10*26
		Theta1_grad += (delta_2.T @ a_1) / self.m   #25*401

		# 正则化
		Theta2_grad[:, 1:] += self.ilambda / self.m * self.Theta2[:, 1:]
		Theta1_grad[:, 1:] += self.ilambda / self.m * self.Theta1[:, 1:]

		# 将梯度展开为向量
		grad = np.vstack((Theta1_grad.reshape(Theta1_grad.size, 1),
							Theta2_grad.reshape(Theta2_grad.size, 1)))
		return J, grad
				
	def feed_forward (self):
		""" 前向传播, 计算hx与代价 """
		#Step 1 前向传播计算h(x)
		a_1 = np.hstack((np.ones((self.m, 1)), self.X))             #5000*401
		z_2 = a_1 @ self.Theta1.T                                   #5000*25
		a_2 = np.hstack((np.ones((self.m, 1)), self.sigmoid(z_2)))  #5000*26
		z_3 = a_2 @ self.Theta2.T                                   #5000*10
		h_x = self.sigmoid(z_3)                                     #5000*10

		#Step 2 计算代价
		J = (- 1 / self.m * np.sum(np.sum(np.eye(self.m) * (self.y @ np.log(h_x.T) 
				+ (1 - self.y) @ np.log(1 - h_x.T)))) 
				+ self.ilambda / (2 * self.m) * (np.sum(np.sum(np.power(self.Theta1[:, 1:], 2))) 
				+ np.sum(np.sum(np.power(self.Theta2[:, 1:], 2)))))
		return a_1, z_2, a_2, h_x, J

	def back_propagation(self, a_1, z_2, a_2, h_x):
		""" 反向传播计算梯度 """
		Theta1_grad = np.zeros(self.Theta1.shape)
		Theta2_grad = np.zeros(self.Theta2.shape)

		#Step 3 反向传播 之 计算输出层的偏导数
		delta_3 = h_x - self.y          #5000*10 Theta1 25*401 Theta2 10*26

		#Step 4 反向传播 之 计算对隐藏层输入z的偏导数
		delta_2 = (delta_3 @ self.Theta2)[:, 1:] * self.sigmoid_gradient(z_2)    #5000*25

		#Step 5 反向传播 之 计算参数的偏导数
		Theta2_grad += (delta_3.T @ a_2) / self.m   #10*26
		Theta1_grad += (delta_2.T @ a_1) / self.m   #25*401

		#正则化
		Theta2_grad[:, 1:] += self.ilambda / self.m * self.Theta2[:, 1:]
		Theta1_grad[:, 1:] += self.ilambda / self.m * self.Theta1[:, 1:]

		return Theta1_grad, Theta2_grad

	def gradient_descent(self, alpha=0.01, num_iters=50):
		""" 梯度下降 """
		for iter in range(num_iters):		
			# 前向传播
			a_1, z_2, a_2, h_x, J = self.feed_forward()
			# 反向传播
			Theta1_grad, Theta2_grad = self.back_propagation(a_1, z_2, a_2, h_x)
			# 梯度下降
			# 不用除以m 之前除以m是因为计算梯度的时候没有考虑进去
			# θ = θ - α*J(θ)偏导数
			self.Theta1 -= alpha * Theta1_grad		
			self.Theta2 -= alpha * Theta2_grad	
			if iter == 0:
				continue
			print('第{}次迭代, 代价值为:{:.6e}'.format(iter, J))	
		# 迭代完成再次计算代价	
		a_1, z_2, a_2, h_x, J = self.feed_forward()
		print('第{}次迭代, 代价值为:{:.6e}'.format(num_iters, J))
		return J	

	def mini_batch_gradient_descent(self, b=2, alpha=0.01, num_iters=10):
		""" Mini_Batch梯度下降 """
		Theta1_grad = np.zeros(self.Theta1.shape)
		Theta2_grad = np.zeros(self.Theta2.shape)

		for iter in range(num_iters):		
			for i in range(0, self.m, b):
				# 每遍历m个样本进行更新参数
				
				# Step 1 取b个样本前向传播计算h(x)
				a_1 = np.hstack((np.ones((b, 1)), self.X[i:i+b, :]))	    # b*401
				z_2 = a_1 @ self.Theta1.T                                   # b*25
				a_2 = np.hstack((np.ones((b, 1)), self.sigmoid(z_2)))  		# b*26
				z_3 = a_2 @ self.Theta2.T                                   # b*10
				h_x = self.sigmoid(z_3)                                     # b*10
				
				# Step 2 计算代价
				J = (- 1 / b * np.sum(np.sum(np.eye(b) * (self.y[i:i+b] @ np.log(h_x.T) 
						+ (1 - self.y[i:i+b]) @ np.log(1 - h_x.T)))) 
						+ self.ilambda / (2 * b) * (np.sum(np.sum(np.power(self.Theta1[:, 1:], 2))) 
						+ np.sum(np.sum(np.power(self.Theta2[:, 1:], 2)))))
				print('第 {} 个样本, 代价值为:{:.6e}'.format(i, J))	
				
				# Step 3 反向传播 之 计算输出层的偏导数
				delta_3 = h_x - self.y[i:i+b]          # b*10 Theta1 25*401 Theta2 10*26
		
				# Step 4 反向传播 之 计算对隐藏层输入z的偏导数
				delta_2 = (delta_3 @ self.Theta2)[:, 1:] * self.sigmoid_gradient(z_2)    #5000*25
		
				# Step 5 反向传播 之 计算参数的偏导数
				Theta2_grad += (delta_3.T @ a_2) / self.m   #10*26
				Theta1_grad += (delta_2.T @ a_1) / self.m   #25*401
		
				# 正则化
				Theta2_grad[:, 1:] += self.ilambda / self.m * self.Theta2[:, 1:]
				Theta1_grad[:, 1:] += self.ilambda / self.m * self.Theta1[:, 1:]

			    # 不用除以m 之前除以m是因为计算梯度的时候没有考虑进去
			    # θ = θ - α*J(θ)偏导数
				self.Theta1 -= alpha * Theta1_grad		
				self.Theta2 -= alpha * Theta2_grad	


