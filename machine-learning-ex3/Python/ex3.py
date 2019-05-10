import numpy as np
import scipy.io as sio
import one_vs_all_functions as ovaf
from logistic_regression import LogisticRegression as lr

## =========== Part 1: 加载以及可视化数据 =============
#
print('Loading and Visualizing Data ...\n')

## 设置参数, 输入灰度图片的像素, 输出为1-10(10代表0)
input_layer_size  = 400   # 20x20 Input Images of Digits		真的是400个神经元
num_labels = 10           # 10 labels, from 1 to 10				输出神经元
                          # (note that we have mapped "0" to label 10)

load_fn = 'ex3data1.mat'            # 读入mat文件返回的是一个字典
load_data = sio.loadmat(load_fn)

# 获取训练集与标记
X = load_data['X']
y = load_data['y']
# 训练集大小
m = X.shape[0]

# 随机获取样本的100行, 进行可视化
rand_indices = np.arange(m)
# shuffle返回None 就地打乱
np.random.shuffle(rand_indices)
sel = X[rand_indices[0:100], :]

ovaf.displayData(sel)


## ============ Part 2a: 使用向量化的逻辑回归 ============
#
print('\nTesting compute_cost() with regularization');

theta_t = np.array([[-2], [-1], [1], [2]])
X_t = np.hstack((np.ones((5,1)), np.arange(1, 16).reshape(3,5).T/10))

y_t = np.array([[1], [0], [1], [0], [1]])
lambda_t = 3

# 建立模型
my_lr = lr(X_t, y_t)
# 计算代价值
J, grad = my_lr.compute_cost(theta_t, lambda_t)

print('Cost: {:0.6f}'.format(float(J)))
print('Expected cost: 2.534819')
print('Gradients:')
for g in grad:	
	print(' {:0.6f}'.format(float(g)))
print('Expected gradients:');
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

## ============ Part 2b: One-vs-All Training ============
#
print('\nTraining One-vs-All Logistic Regression...\n')

ilambda = 0.1
alpha = 0.03
num_iters = 1000
all_theta = ovaf.one_vs_all(X, y, alpha, ilambda, num_labels, num_iters)
# 将参数保存到文件
sio.savemat('parameter.mat', {'all_theta': all_theta, 'lambda': ilambda})

## ================ Part 3: 计算精度 ================
load_data = sio.loadmat('parameter.mat')
all_theta = load_data['all_theta']
pred = ovaf.predict_one_vs_all(X, all_theta)

print('\nTraining Set Accuracy: {:0.2f}%'.format(float(pred[pred == y].size / y.size * 100)))


