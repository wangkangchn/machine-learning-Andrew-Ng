"""
			
				PCA
		

"""

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib import image as pimg
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pcaFunctions as pcaf
import kMeansFunctions as kmf

# ~ ## ================== Part 1: 加载示例数据集  ===================
# ~ #

# ~ print('可视化示例数据集\n');

# ~ load_data= sio.loadmat ('ex7data1.mat')
# ~ X = load_data['X']

# ~ # 打开交互模式
# ~ plt.ion()

# ~ plt.plot(X[:, 0], X[:, 1], 'bo')
# ~ plt.axis([0.5, 6.5, 2, 8]) 


# ~ ## =============== Part 2: PCA(主成分分析法) ===============
# ~ #

# ~ print('在示例数据集上运行PCA\n');

# ~ # 运行PCA的第一步是进行特征的均值归一化
# ~ X_norm, mu, sigma = pcaf.featureNormalize(X)

# ~ # 运行PCA, 返回特征向量, 对角矩阵
# ~ U, S = pcaf.pca(X_norm)

# ~ # 画出以数据均值为中心的特征向量。这些线表示数据集中最大变化的方向。

# ~ pcaf.draw_line(mu, mu + 1.5 * S[0] * U[:,0].T, '-k', LineWidth=2)
# ~ pcaf.draw_line(mu, mu + 1.5 * S[1] * U[:,1].T, '-k', LineWidth=2)

# ~ print('Top eigenvector: ');
# ~ print(' U[:, 0] = {} {}'.format(U[0, 0], U[1, 0]))
# ~ print('(you should expect to see -0.707107 -0.707107)\n');


# ~ ## =================== Part 3: 降维 ===================
# ~ #
# ~ print('对示例数据集进行降维处理\n');

# ~ # 绘制规范化后的数据集
# ~ plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo');
# ~ plt.axis([-4, 3, -4, 3,])

# ~ # 将为1维
# ~ K = 1
# ~ Z = pcaf.project_data(X_norm, U, K)
# ~ print('Projection of the first example: {}'.format(Z[0]))
# ~ print('\n(this value should be about 1.481274)\n');

# ~ X_rec  = pcaf.recover_data(Z, U, K)
# ~ print('Approximation of the first example: {} {}'.format(X_rec[0, 0], X_rec[0, 1]))
# ~ print('\n(this value should be about  -1.047419 -1.047419)\n');

# ~ # 连绘制接投影点和原始点的线
# ~ # 绘制投影点
# ~ plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
# ~ # 绘制连接线
# ~ for i in range(X_norm.shape[0]):
    # ~ pcaf.draw_line(X_norm[i,:], X_rec[i,:], '--k', LineWidth=1)

# ~ plt.ioff()
# ~ plt.show()

# ~ ## =============== Part 4: 加载以及可视化人脸数据集 =============
# ~ #
# ~ print('\nLoading face dataset.\n');

# ~ # 开启交互模式显示图片
# ~ plt.ion()

# ~ # 加载加载人脸数据集
# ~ load_data = sio.loadmat('ex7faces.mat')
# ~ X = load_data['X']

# ~ # 显示前100个图像
# ~ plt.subplot(2, 2, 1)
# ~ pcaf.display_data(X[:100, :])


# ~ ## =========== Part 5: 使用PCA进行图像的压缩(仅保留K个主成分)  ===================
# ~ #
# ~ print('Running PCA on face dataset.')
# ~ print('(this might take a minute or two ...)\n')

# ~ # 均值归一化
# ~ X_norm, mu, sigma = pcaf.featureNormalize(X)

# ~ # 运行PCA
# ~ U, S = pcaf.pca(X_norm)

# ~ # 可视化前36个主成分
# ~ plt.subplot(2, 2, 2)
# ~ pcaf.display_data(U[:, :36].T)		# U包含所有的主成分


# ~ ## ============= Part 6: 减少人脸图像集的维度 =================
# ~ # 

# ~ print('Dimension reduction for face dataset.\n')

# ~ # 选择K(保留99%的方差)
# ~ K = pcaf.select_K(S)
# ~ Z = pcaf.project_data(X_norm, U, K)

# ~ print('The original data X has a size of: {}'.format(X.shape))
# ~ print('The projected data Z has a size of: {}'.format(Z.shape))

# ~ ## ==== Part 7: 可视化使用PCA后人脸图像 ====
# ~ #

# ~ print('Visualizing the projected (reduced dimension) faces.\n\n');

# ~ K = pcaf.select_K(S)
# ~ X_rec  = pcaf.recover_data(Z, U, K)

# ~ # 显示规范化的数据
# ~ plt.subplot(1, 2, 1)
# ~ pcaf.display_data(X_norm[:100, :])
# ~ plt.title('Original faces');
# ~ plt.axis('off')

# ~ # 显示原始的数据	规范化后还是有差异
# ~ plt.figure()
# ~ pcaf.display_data(X[:100, :])
# ~ plt.title('X faces');
# ~ plt.axis('off')

# ~ # 显示重构后的人脸
# ~ plt.subplot(1, 2, 2)
# ~ pcaf.display_data(X_rec[:100, :])
# ~ plt.title('Recovered faces')
# ~ plt.axis('off')

# ~ plt.ioff()
# ~ plt.show()

# ~ ########################################################################

# ~ print('不同的压缩维度对图像的损失情况.\n')
# ~ print('\nLoading face dataset.\n');
# ~ # 开启交互模式显示图片
# ~ plt.ion()

# ~ # 加载加载人脸数据集
# ~ load_data = sio.loadmat('ex7faces.mat')
# ~ X = load_data['X']

# ~ print('Running PCA on face dataset.')
# ~ print('(this might take a minute or two ...)\n')

# ~ # 均值归一化
# ~ X_norm, mu, sigma = pcaf.featureNormalize(X)
# ~ # 运行PCA
# ~ U, S = pcaf.pca(X_norm)

# ~ accuracy = 0.99
# ~ for i in range(10):
	# ~ # 选择K(保留99%的方差)
	# ~ K = pcaf.select_K(S, accuracy-0.1*i)
	# ~ # 压缩
	# ~ Z = pcaf.project_data(X_norm, U, K)	
	# ~ # 恢复
	# ~ X_rec  = pcaf.recover_data(Z, U, K)
	
	# ~ # 显示重构后的人脸
	# ~ plt.figure()
	# ~ pcaf.display_data(X_rec[:100, :])
	# ~ plt.title('Recovered faces (K: {} accuracy: {})'.format(K, accuracy-0.1*i))
	# ~ plt.axis('off')

# ~ plt.ioff()
# ~ plt.show()

########################################################################

## === Part 8(a): K-Means 与 PCA 一起使用 ===

# 加载图片
A = pimg.imread('bird_small.png')

img_size = A.shape

# 竖向排列 第一层红 第二层绿 第三次蓝
X = A.reshape(img_size[0] * img_size[1], 3)	  
# K-Means 聚簇分类 
K = 16 
max_iters = 10
initial_centroids = kmf.kMeans_init_centroids(X, K)
centroids, idx = kmf.run_kMeans(X, initial_centroids, max_iters)

# 随机选择1000个样本进行处理
sel = np.floor(np.random.rand(1000, 1) * X.shape[0]) + 1

# 创建颜色地图映射
# ~ palette = hsv(K);
# ~ colors = palette(idx(sel), :);

# Visualize the data and centroid memberships in 3D
fig = plt.figure()                      
ax = Axes3D(fig)
sel = sel.astype(np.int16)
# 绘制三维散点图
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2])

plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships');
plt.show()

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===

X_norm, mu, sigma = pcaf.featureNormalize(X)

# PCA and project the data to 2D
U, S = pcaf.pca(X_norm);
Z = pcaf. project_data(X_norm, U, 2);
# Plot in 2D
plt.figure()
kmf.plot_data_points(Z[sel[:, 0], :], idx[sel[:, 0]], K)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');

plt.show()









