"""
	
				K-Means 分类算法
		
		1. 分配 find_closest_centroids
		2. 移动	compute_centroids		

"""
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib import image as pimg 		# mpimg 用于读取图片
import kMeansFunctions as kmf

## ================= Part 1: 划分聚簇 ====================
#
print('划分聚簇...\n')

# 加载数据
load_data = sio.loadmat('ex7data2.mat')
X = load_data['X']

# 设置初始化的聚簇中心
K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# 划分聚簇
idx = kmf.find_closest_centroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: \n')
print(' {} {} {}'.format(int(idx[0]), int(idx[1]), int(idx[2])))
print('\n(the closest centroids should be 0, 2, 1 respectively)\n');


## ===================== Part 2: 移动聚簇中心 =========================
# 

print('移动聚簇中心.\n');

centroids = kmf.compute_centroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: ')
print(centroids)
print('\n(the centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]\n')


# ~ ## =================== Part 3: 运行K-Means算法 ======================
# ~ #
print('\nRunning K-Means clustering on example dataset.\n');

# 加载实例数据集
load_data = sio.loadmat('ex7data2.mat')
X = load_data['X']

# 设置聚簇个数, 以及K-Means算法的迭代次数
K = 3
max_iters = 10

initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

centroids, idx = kmf.run_kMeans(X, initial_centroids, max_iters, True)
print('K-Means Done.\n')


# ~ ## ============= Part 4: 对图片使用K-Means进行压缩 ===============
# ~ #

print('对图片使用K-Means进行压缩\n');

# pimg直接读取图片为numpy数组格式
A = pimg.imread('bird_small.png')
# 获取图片尺寸
img_size = A.shape

# 竖向排列 第一层红 第二层绿 第三次蓝
X = A.reshape(img_size[0] * img_size[1], 3)	       

# K-Means参数
K = 16 
max_iters = 10

# 初始化距聚类中心
initial_centroids = kmf.kMeans_init_centroids(X, K)
# 运行K-Means进行分类
centroids, idx = kmf.run_kMeans(X, initial_centroids, max_iters)


# ~ ## ================= Part 5: 图片压缩 ======================
# ~ #

print('应用K-Means进行图片压缩.\n');
# 分完聚簇之后, 再找成员? 对对对, 因为在K-Means算法中, 每次迭代是先分再移动, 
# 所以最后一次的中心是没有进行重新分配的
idx = kmf.find_closest_centroids(X, centroids); 

# 进行图像的恢复
# folat 类型不能作索引, 所以进行数组作为索引的时候, 需要先转换类型为int
idx = idx.astype(np.int16)
X_recovered = centroids[idx[:, 0], :]
# 恢复图片的形状已进行显示
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(A)
plt.axis('off')
plt.title('Original')

# 显示压缩后的图片
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.axis('off')
plt.title('Compressed, with {} colors.'.format(K))
plt.show()
