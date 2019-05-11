""" 

				神经网络相关函数
				
	函数功能说明:
			
		display_data					-	可视化数据
		rand_initialize_weights		- 	随机初始化参数theta(-epsilon ~ +epsilon)
		debug_initialize_weights		-	产生随机的调试数据
		compute_numerical_gradient	-	计算近似的数值计算梯度
		check_NNGradients			-	梯度检验
		sigmoid						-   激活函数
		predict						-	利用学习得到的参数进行预测
		
		
"""
import numpy as np
from matplotlib import pyplot as plt
from neuralNetwork import NeuralNetwork as nn

def display_data(X, example_width=None):
    """ 可视化数据--显示灰度照片 """
    # Set example_width automatically if not passed in
    if example_width is None or example_width == 0:
        example_width = int(round(np.sqrt(X.shape[1])))        #round四舍五入

    #Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    #Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    #Between images padding
    pad = 1;

    #Setup blank display
    display_array = - np.ones((int(pad + display_rows * (example_height + pad)), 
                           int(pad + display_cols * (example_width + pad))))

    # 将要显示的图片通通打入显示数组中
    curr_ex = 0
    for j in range(0, display_rows):
        for i in range(0, display_cols):
            if curr_ex > m-1:   #与Octave的区别 下标从1开始
                break

            # Copy the patch
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, :]))

            #每个图片的其实行列下标
            row_start = pad + j * (example_height + pad)
            col_start = pad + i * (example_width + pad)
            #使用切片对每一个图像区域的每个值进行赋值!!!
            display_array[row_start:row_start+example_height,
                          col_start:col_start+example_width] = X[curr_ex, :].reshape(example_height, example_width) / max_val
            curr_ex = curr_ex + 1
        if curr_ex > m-1:
            break

    #Display Image
    plt.imshow(display_array.T, interpolation='bicubic')
    # Do not show axis
    plt.xticks([]), plt.yticks([])
    plt.show()

def rand_initialize_weights(L_in, L_out):
	""" 初始化神经网络的参数 0-1 """
	W = np.zeros((L_out, 1 + L_in))
		
	#是对每一层的theta分别初始化, 可以使用相同的epsilon_init, 也可使用不同的
	epsilon_init = np.sqrt(6 / (L_in + L_out))
	# ~ epsilon_init = 0.12;
	#~ epsilon_init = sqrt(6 / (L_in + L_out));
	W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
	
	return W

def debug_initialize_weights(fan_out, fan_in):
	""" 使用同一种方式生成参数以及训练样本 """
	W = np.zeros((fan_out, 1 + fan_in))
	
	# 使用sin函数进行初始化
	W = np.sin(np.arange(1, W.size+1)).reshape(W.shape) / 10
	return W
	
def compute_numerical_gradient(nn, theta):
	""" 进行近似的数值梯度计算 """        	
	numgrad = np.zeros((theta.size, 1))
	perturb = np.zeros((theta.size, 1))
	e = 1e-4
	# 已经将参数展成了一个长向量, 对每一个参数进行梯度检查, 检查时仅修改该参数, 其余参数不变
	for p in range(theta.size):
	    # 设置偏移值    
	    perturb[p] = e
	    loss1, _ = nn.cost_function(theta - perturb)
	    loss2, _ = nn.cost_function(theta + perturb)
	    # 计算数值梯度
	    numgrad[p] = (loss2 - loss1) / (2*e)
	    perturb[p] = 0	# 恢复避免扰乱其他的值计算
	return numgrad

def check_NNGradients (ilambda=0):
	""" 
	
		使用一个小型神经网络进行梯度检测, 比较分析梯度与数值梯度的差距
	只进行一次传播就行, 只是验算是否正确, 不需要进行梯度下降, 训练时才需要
	
	"""	
	# 神经网络参数
	input_layer_size = 3;
	hidden_layer_size = 5;
	num_labels = 3;
	m = 5;
	
	# 随机生成参数
	Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
	Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
	# 随机生成训练集 m*input_layer_size
	X  = debug_initialize_weights(m, input_layer_size - 1)
	y  = np.mod(np.arange(0, m), num_labels).reshape(m, 1)

	# 展开参数
	nn_params = np.vstack((Theta1.reshape(Theta1.size, 1), Theta2.reshape(Theta2.size, 1)))
	my_nn = nn(nn_params, input_layer_size, hidden_layer_size,
			num_labels, X, y, ilambda)

	# 进行分析梯度计算
	cost, grad = my_nn.cost_function()	
	# 进行数值梯度计算
	numgrad = compute_numerical_gradient(my_nn, nn_params)

	# 显示分析计算的梯度以及近似计算得到的数值梯度
	for num, analyze in zip(numgrad, grad):		# zip可以使多个迭代器一起迭代
		print('{:.6f} {:.6f}'.format(float(num), float(analyze)))
	print('\n上面的两列应该非常的相近.\n' 
	        '(左-数值梯度, 右-解析梯度)\n')
	
	# Evaluate the norm of the difference between two solutions.  
	# If you have a correct implementation, and assuming you used EPSILON = 0.0001 
	# in compute_numerical_gradient.m, then diff below should be less than 1e-9
	diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
	
	print('如果你的反向传播算法是正确的, 相对偏差会非常的小(小于1e-9) \n' 
	        '相对偏差: {:.6e}'.format(diff))

def sigmoid(t):	
	return 1 / (1 + np.exp(-t))	
	
def predict(Theta1, Theta2, X):
	""" 进行神经网络的预测 """

	m = X.shape[0]
	num_labels = Theta2.shape[0]
	
	# 隐藏单元激活值
	h1 = sigmoid(np.hstack((np.ones((m, 1)), X)) @ Theta1.T)
	# 输出层概率
	h2 = sigmoid(np.hstack((np.ones((m, 1)), h1)) @ Theta2.T)
	# 预测值
	p = np.argmax(h2, 1).reshape(m, 1)
	
	return p

