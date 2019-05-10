"""
                多元分类器---逻辑回归

		就是训练多个二元分类器, 对每个样本使用这些分类器, 输出结果最大的
	即为我们所需要的标签

"""
import math
import numpy as np
from matplotlib import pyplot as plt
from logistic_regression import LogisticRegression as LR

def one_vs_all (X, y, alpha, ilambda, num_labels, num_iters):
    """
        训练多元分类器, 返回所有分类器的参数, 到all_theta中每一行代表一个分类
    器的参数
    """
    # 获取训练集大小以及特征数量
    m, n = X.shape

    # 保存所有参数, 加入偏置单元
    all_theta = np.zeros((num_labels, n+1))
    X_t = np.hstack((np.ones((m, 1)), X))
    y_temp = np.zeros(y.shape)
    
    # 训练分类器
    for c in range(1, num_labels+1):
        # 重新打定标签      
        y_temp[y==c] = 1
        y_temp[y!=c] = 0     
        classifier = LR(X_t, y_temp)
        classifier.gradient_descent(alpha, ilambda, num_iters)
       
        # 将theta转为二维行向量进行存储
        all_theta[c-1, :] = classifier.theta.reshape(1, n+1)
        print('classifier {} | Cost: {:0.6e}'.format(c, float(classifier.J_end)))
	
    return all_theta

def sigmoid (z): 
	return 1 / (1 + np.exp(-z))
	
def predict_one_vs_all (X, all_theta):
    """
    
    对传入的X进行预测, 返回p, p为包含样本预测标记的列向量
    
    """
    # 获取样本(每行)在各个分类器上的最大值的下标, 注意, 这里进行下标+1是为了与
    # 课件下标进行匹配, 从1开始
    # 添加偏置单元
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    p = np.argmax(sigmoid(X @ all_theta.T), 1) + 1

    # 转变为二维列向量
    p = p.reshape(p.size, 1)
    return p

def predict_nn(Theta1, Theta2, X):
	m = X.shape[0]
	num_labels = Theta2.shape[0]

	# ~ # 添加第一层偏置单元
	# ~ X_1 = np.hstack((np.ones((m, 1)), X))
	# ~ # 进行隐藏层神经单元的计算
	# ~ X_2 = sigmoid(X_1 @ Theta1.T)

	# ~ # %添加第二层偏置单元
	# ~ X_2 = np.hstack((np.ones((m, 1)), X_2))
	# ~ # %进行输出层神经单元的计算
	# ~ h_x = sigmoid(X_2 @ Theta2.T)

	# ~ # 统计预测值, argmax返回一维向量, 需要将其转换为二维向量
	# ~ p = (np.argmax(h_x, 1) + 1).reshape(m,1)

	p = (np.argmax(sigmoid(np.hstack((np.ones((m, 1)),
			sigmoid(np.hstack((np.ones((m, 1)), X)) @ Theta1.T))) @ Theta2.T), 1)+1).reshape(m,1)
	return p

def display_data(X, example_width=None):
    """ 可视化数据--显示灰度照片 """
    # Set example_width automatically if not passed in
    if example_width is None or example_width == 0:
        example_width = int(round(np.sqrt(X.shape[1])))        # round四舍五入

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1;

    # Setup blank display
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
    plt.imshow(display_array.T, interpolation='bicubic', cmap='gray')
    # Do not show axis
    plt.xticks([]), plt.yticks([])
    plt.show()
