""" 
			
			协同过滤作业文件
	
""" 
import numpy as np
import scipy.io as sio
import collaborativeFilteringFunctions as cff

## =============== Part 1: Loading movie ratings dataset ================

print('Loading movie ratings dataset.\n')

# 加载数据
load_data = sio.loadmat ('ex8_movies.mat')

Y = load_data['Y']	# Y is a 1682x943 matrix
R = load_data['R']	# R is a 1682x943 matrix


# 提取有用的分进行取平均值 R(1,:)是一个逻辑数组
print('Average rating for movie 1 (Toy Story): {:.6f} / 5\n\n'.format(np.mean(Y[0, R[0, :]==1])))	


## ============ Part 2: 协同过滤代价函数 ===========
#

# 加载测试参数
load_data = sio.loadmat('ex8_movieParams.mat')
X = load_data['X']
Theta = load_data['Theta']
# ~ for key, value in load_data.items():
	# ~ print('key: {} \n value: {}\n\n'.format(key, value))

# 设置参数
num_users = 4
num_movies = 5
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

params = np.vstack((X.reshape(X.size, 1), Theta.reshape(Theta.size, 1)))
# ~ # 计算代价值
J, _ = cff.cofi_costfunc(params, Y, R, num_users, num_movies, num_features, 0)
           
print('Cost at loaded parameters: {:.6f} '.format(J))
print('(this value should be about 22.22)')


## ============== Part 3: 计算梯度 ==============
#
print('Checking Gradients (without regularization)... ');

cff.check_cost_function()


## ========= Part 4: 正则代价值 ========
#

J, _ = cff.cofi_costfunc(params, Y, R, num_users, num_movies, num_features, 1.5)
           
print('Cost at loaded parameters (lambda = 1.5): {} '.format(J))
print('this value should be about 31.34)')


## ======= Part 5: 正则梯度检测 ======
#
 
print('\nChecking Gradients (with regularization) ... \n');

cff.check_cost_function(1.5)



