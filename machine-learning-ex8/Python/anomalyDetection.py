"""

				异常检查

"""
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import anomalyDetectionFunctions as adf

## ================== Part 1: 加载实例实例数据集  ===================
#
print('Visualizing example dataset for outlier detection.\n\n');

# 加载数据
load_data = sio.loadmat('ex8data1.mat')
X = load_data["X"]
Xval = load_data["Xval"]
yval = load_data["yval"]

# 开启交互显示模式
plt.ion()

# 可视化数据集
plt.plot(X[:, 0], X[:, 1], 'bx')
plt.axis([0, 30, 0, 30])
plt.title('Example Dataset')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')


## ================== Part 2: 估计模型所需要的统计信息 ===================
#
print('Visualizing Gaussian fit.\n\n');

# 估计均值mu和方差sigma2
mu, sigma2 = adf.estimate_gaussian(X)

# 建立高斯概率分布模型
p = adf.multivariate_gaussian(X, mu, sigma2)	

# 可视化高斯模型拟合的数据
adf.visualize_fit(X,  mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')

## ================== Part 3: 查找离散点 ===================
#

# 交叉验证集
pval = adf.multivariate_gaussian(Xval, mu, sigma2)

epsilon, F1 = adf.select_threshold(yval, pval)
print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('Best F1 on Cross Validation Set: {}'.format(F1))
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)\n');

# 查找数据集上的离散点, 并绘制出来
outliers = np.where(p < epsilon)[0]

plt.plot(X[outliers, 0], X[outliers, 1], 'ro', MarkerSize=8)

# 关闭交互显示模式
plt.ioff()
plt.show()

## ================== Part 4: Multidimensional Outliers ===================
#

# 加载数据
load_data = sio.loadmat('ex8data2.mat')
X = load_data["X"]
Xval = load_data["Xval"]
yval = load_data["yval"]

mu, sigma2 = adf.estimate_gaussian(X)

# 计算训练集概率值
p = adf.multivariate_gaussian(X, mu, sigma2)

# 计算交叉验证集概率值
pval = adf.multivariate_gaussian(Xval, mu, sigma2)

# 选择最佳的阈值
epsilon, F1 = adf.select_threshold(yval, pval)

print('Best epsilon found using cross-validation: {}'.format(epsilon))
print('Best F1 on Cross Validation Set: {}'.format(F1))
print('   (you should see a value epsilon of about 1.38e-18)')
print('   (you should see a Best F1 value of 0.615385)')
print('# Outliers found: {}'.format(p[p < epsilon].size))
