""" 
                神经网络测试文件 
        
			进行识别手写数字的训练
        
"""
import numpy as np

import scipy.io as sio
from neuralNetwork import NeuralNetwork as nn
import neuralNetworkFunctions as nnf

# 设置神经网络参数
input_layer_size  = 400     # 400个输入单元(输入 20x20像素 数字图片)
hidden_layer_size = 25      # 25 个隐藏单元
num_labels = 10             # 10 个输出单元(10个标签)  

# =========== Part 1: 加载以及可视化数据 =============     
#
#加载训练集
load_fn = 'ex4data1.mat'
load_data = sio.loadmat(load_fn)

X = load_data['X']
y = load_data['y']
y[y==10] = 0                # 将标记 10 转换为 0
m = y.shape[0]              # 训练集大小          

rand_indices = np.arange(m)	# 随机获取样本的100行, 进行可视化
np.random.shuffle(rand_indices)	# shuffle返回None 就地打乱
sel = X[rand_indices[0:100], :]

# 显示数据集
nnf.display_data(sel)


## ================ Part 2: 加载参数 ================
#
print('加载神经网络参数...\n')

# 加载参数
load_theta = sio.loadmat('ex4weights.mat')
Theta1 = load_theta['Theta1']

# 将参数的最后一行添加到开始位置
load_Theta2 = load_theta['Theta2']
q = load_Theta2[-1]
Theta2 = np.vstack((q.reshape(1, q.shape[0]), load_Theta2[:-1]))

# 将参数展开为一个长向量
nn_params = np.vstack((Theta1.reshape(Theta1.size, 1), Theta2.reshape(Theta2.size, 1)))


## ================ Part 3: 计算代价 (feed_forward) ================
#
print('利用神经网络的前向传播计算代价...\n')

# 正则参数设为0, 即代表不进行正则化
ilambda = 0

# 创建神经网络
my_nn = nn(nn_params, input_layer_size, hidden_layer_size,
			num_labels, X, y, ilambda)
			
_,_,_,_,J = my_nn.feed_forward()
print('''从ex4weights中加载的参数计算得到的代价值为: {:.6f} 
(这个值大约为: 0.287629)\n'''.format(J))


## =============== Part 4: 实现正则化 ===============
#
print('检查正则代价... \n')

#使用lambda来进行正则化的区分
ilambda = 1	

my_nn_rec = nn(nn_params, input_layer_size, hidden_layer_size,
			num_labels, X, y, ilambda)
			
_,_,_,_,J = my_nn_rec.feed_forward()
print('''从ex4weights中加载的参数计算得到的代价值为: {:.6f} 
(这个值大约为: 0.383770)\n'''.format(J))


## ================ Part 5: Sigmoid Gradient  ================
#
print('检查sigmoid的导数...\n')

g = my_nn_rec.sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient 评估 [-1 -0.5 0 0.5 1]:')
v=''
for value in g:
	v += '{:.4f}  '.format(value)
print(v)


## ================ Part 6: 初始化参数 ================
#
print('\n初始化神经网络的参数...')

initial_Theta1 = nnf.rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = nnf.rand_initialize_weights(hidden_layer_size, num_labels)

# 将参数展成一个长向量
initial_nn_params = np.vstack((initial_Theta1.reshape(initial_Theta1.size, 1), 
								initial_Theta2.reshape(initial_Theta2.size, 1)))


## =============== Part 7: 实现反向传播 ===============
#
print('非正则梯度检测 ... \n')

nnf.check_NNGradients()


## =============== Part 8: 实现正则梯度检测 ===============
#
print('\n正则梯度检测 ... \n')

ilambda = 3
nnf.check_NNGradients(ilambda)

my_nn_rec = nn(nn_params, input_layer_size, hidden_layer_size,
			num_labels, X, y, ilambda)
			
_,_,_,_,debug_J = my_nn_rec.feed_forward()

print('\n固定调试的代价值 (lambda = {}): {:.6f} '
         '\n(对于 lambda = 3, 这个值应该大约是 0.576051)\n'.format(ilambda, debug_J))


## =================== Part 9: 训练神经网络 ===================
#
print('\n训练神经网络... \n')

#训练参数
ilambda = 1
alpha = 1
num_iters = 500

#创建神经网络
train_nn = nn(initial_nn_params, input_layer_size, hidden_layer_size,
			num_labels, X, y, ilambda)
#使用梯度下降进行训练
cost = train_nn.gradient_descent(alpha, num_iters)

#获得训练参数
Theta1 = train_nn.Theta1
Theta2 = train_nn.Theta2

# 保存训练参数
save_fn = 'myNeuralNetworkTheta_{}.mat'.format(num_iters)
try:
	print('保存训练参数...')
	sio.savemat(save_fn, {'Theta1':Theta1, 'Theta2':Theta2})
except Exception as e:
	print("Error: {}".format(e))
else:
	print('参数保存成功...')
	
	
## ================= Part 10: 可视化权重参数 =================
# “可视化”神经网络正在学习什么, 显示隐藏单元, 查看它们在数据中捕获哪些特
#
print('\n可视化神经网络隐藏层... \n')

nnf.display_data(Theta1[:, 1:])


## ================= Part 11: 进行预测精度的计算 =================
#

pred = nnf.predict(Theta1, Theta2, X);
p = float(pred[pred==y].shape[0] / m) * 100
print('训练集精度: {:.3f}%\n'.format(p))

msg = 'ilambda = {} alpha = {} num_iters = {}   predict = {:.3f}%\n'.format(
		ilambda, alpha, num_iters, p)

try:
	with open('nn_predict.txt', 'a') as f:
		f.write(msg)
except Exception as e:
	print("Error: {}".format(e))



