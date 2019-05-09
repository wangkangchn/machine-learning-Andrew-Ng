"""
				逻辑回归测试文档

"""
import numpy as np
from matplotlib import pyplot as plt 
from logistic_regression import LogisticRegression
import logistic_regression_functions as lrf

# 加载数据
data = np.loadtxt('ex2data2.txt', delimiter=',') 

m = data.shape[0]	# 训练集大小
X = data[:, 0:2]	# 成绩	
y = data[:, -1:]	# 标记

# 进行多项式的映射
X_m = lrf.map_feature(X[:, 0:1], X[:, 1:2])

# 设置参数
alpha = 1
ilambda = 0
num_iters = 1000

# 创建模型
my_linear = LogisticRegression(X_m, y)

print('\n梯度下降学习中(lambda = {})...'.format(ilambda))
my_linear.gradient_descent(alpha, ilambda, num_iters)

theta = ''
for val in my_linear.theta:
	theta += str(val)[str(val).find('[')+1:-1]
	theta += '\n'
print('θ: \n{}'.format(theta))
print('梯度下降学习结束...')

print('\n模型精度为: {:.2f}%'.format(my_linear.accuracy()*100))

print('\n梯度下降预测: ')
# 预测数据
p = lrf.map_feature(data[0:5, 0:1], data[0:5, 1:2])

value = my_linear.prediction(p)	

for i in range(len(p)):
	print('{} -> 特征1: {:.2f}  特征2: {:.2f}  -->  标记为1的概率: {:.2f} %'.format(
			i+1, p[i][1], p[i][2], float(value[i])*100))	

# 绘制决策边界
my_linear.plot_decision_boundary(data)	
