"""

                PCA相关函数

"""
import numpy as np
from matplotlib import pyplot as plt

def featureNormalize(X):
    """ 特征均值归一化 """
    
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm / sigma
        
    return X_norm, mu, sigma

def pca (X):
    """ PCA """
    m = X.shape[0]
    
    # 创建协方差矩阵
    Sigma = 1 / m * X.T @ X
    # 获取特征向量, 对角矩阵
    U, S, V = np.linalg.svd(Sigma)
    return U, S

def select_K (S, accuracy=0.99):
	""" 选择K(保留99%的方差) """
	# 遍历所有的维度, 计算保留方差
	for K in range(1, S.size+1):
		# ~ print('K: {} Error({}): {}'.format(K, accuracy, np.sum(S[:K]) / np.sum(S)))
		if (np.sum(S[:K]) / np.sum(S)) >= accuracy:
			break
	print('K: {} accuracy: {}'.format(K, accuracy))		
	return K		
	  
def draw_line(p1, p2, *argv, **kargv):
    """ 绘制直线 """
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], *argv, **kargv)

def project_data(X, U, K):
    """ 进行映射降维 """
    Z = X @ U[:, :K] 
    return Z
    
def recover_data(Z, U, K):
    """ 从低维度恢复为高维度 """
    X = Z @ U[:, :K].T
    return X
    
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
    
    # 显示图片
    plt.imshow(display_array.T, interpolation='bicubic', cmap='gray')   
    # 不显示坐标轴
    plt.axis('off') 


