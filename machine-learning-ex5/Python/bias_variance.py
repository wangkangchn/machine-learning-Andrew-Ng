"""
	
			进行线性回归偏差与方差的调试分析---绘制学习曲线
	
	1. 单特征回归分析, 绘制学习曲线
	2. 高阶多项式分析
	3. 自动选择lambda
	4. 泛化误差

"""
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from linear_regression import LinearRegression as LR
import bias_variance_functions as bvf

## =========== Part 1: 加载以及可视化数据 =============
#

print('加载以及可视化数据...\n')

load_fn = 'ex5data1.mat'
load_data = sio.loadmat(load_fn)

# 提取训练集
X = load_data['X']
y = load_data['y']
m = X.shape[0]
# 提取交叉验证集
Xval = load_data['Xval']
yval = load_data['yval']
# 提取测试集
Xtest = load_data['Xtest']
ytest = load_data['ytest']

# 显示训练集
# ~ plt.plot(X, y, 'rx', markersize=8, linewidth=1.5)
# ~ plt.xlabel('Change in water level (x)')
# ~ plt.ylabel('Water flowing out of the dam (y)')
# ~ plt.show()


# =========== Part 2: 计算正则线性回归代价与梯度 =============
#

theta = np.ones((2,1))
ilambda = 1		# 使用iambda控制是否正则化

# 创建模型
my_lr = LR(np.hstack((np.ones((m, 1)), X)) , y)
# 计算代价值与梯度
cost, grad = my_lr.computeCostReg(theta, ilambda)

print('在 theta = [1; 1] 时, 代价值为: {:.6f} '
        '\n(这个值大约是 303.993192)\n'.format(float(cost)))
print('在 theta = [1; 1] 时, 梯度分别为: [{:.6f}; {:.6f}] '
        '\n(这个值大约是 [-15.303016; 598.250744])\n'.format(float(grad[0]), float(grad[1]))) 


## =========== Part 3: 训练线性回归模型 =============
#

ilambda = 0
alpha = 0.001
num_iters = 5000

# 使用梯度下降进行训练
my_lr.gradientDescentReg(alpha, ilambda, num_iters)
theta = my_lr.theta

# 对数据进行拟合
plt.plot(X, y, 'rx', markersize=10, LineWidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, np.hstack((np.ones((m, 1)), X)) @ theta, '--', LineWidth=2)
plt.show()


## =========== Part 4: 绘制线性回归学习曲线 =============
#

print("计算训练误差以及交叉验证误差...")
ilambda = 0;
error_train, error_val = bvf.learningCurve(np.hstack((np.ones((m, 1)), X)), y, 
        np.hstack((np.ones((Xval.shape[0], 1)), Xval)), yval, ilambda)

plt.plot(np.arange(1, m+1), error_train)
plt.plot(np.arange(1, m+1), error_val)

plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show()

print('# 训练样本\t训练误差\t交叉验证误差');
for i in range(m):
    print('  {}\t\t{:.6f}\t{:.6f}'.format(i+1, float(error_train[i]), float(error_val[i])))


## =========== Part 6: 多项式回归 =============
#

# 最高阶
p = 8;

# 对X进行映射以及进行规范化
X_poly = bvf.polyFeatures(X, p)
X_poly, mu, sigma = bvf.featureNormalize(X_poly)  # 规范化
X_poly = np.hstack((np.ones((m, 1)), X_poly))     # 加入 1 列

# 使用 mu sigma 对测试集进行映射规范
X_poly_test = bvf.polyFeatures(Xtest, p)
X_poly_test = (X_poly_test - mu) / sigma
X_poly_test = np.hstack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))     # 加入 1 列

# 使用 mu sigma 对映射集进行映射规范
X_poly_val = bvf.polyFeatures(Xval, p)
X_poly_val = (X_poly_val - mu) / sigma
X_poly_val = np.hstack((np.ones((X_poly_val.shape[0], 1)), X_poly_val))     # 加入 1 列

print('\n规范法训练样本 1:');
print('  {}  '.format(X_poly[1, :]))


## =========== Part 7: 绘制多项式回归学习曲线 =============
#

ilambda = 0.01
m_alpha = 0.001			# 多项式回归学习率
my_lr = LR(X_poly, y)
my_lr.gradientDescentReg(m_alpha, ilambda, num_iters)
theta = my_lr.theta

# 进行数据拟合
print('进行多项式数据拟合...\n')
plt.figure(1)
plt.plot(X, y, 'rx', MarkerSize=10, LineWidth=1.5)
bvf.plotFit(np.min(X), np.max(X), mu, sigma, theta, p)

plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title ('Polynomial Regression Fit (lambda = {})'.format(ilambda))

print('绘制多项式学习曲线...')
plt.figure(2)
error_train, error_val = bvf.learningCurve(X_poly, y, X_poly_val, yval, ilambda)

plt.plot(np.arange(1, m+1), error_train)
plt.plot(np.arange(1, m+1), error_val)

plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(ilambda))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend(['Train', 'Cross Validation'])
plt.show()

print('\nPolynomial Regression (lambda = {})\n'.format(ilambda))
print('\n# 训练样本\t训练误差\t交叉验证误差');
for i in range(m):
    print('  {}\t\t{:.6f}\t{:.6f}'.format(i+1, float(error_train[i]), float(error_val[i])))


## =========== Part 8: 自动选择lambda =============
#

# 计算多项式的训练与验证误差
lambda_vec, error_train, error_val = bvf.validationCurve(X_poly, y, X_poly_val, yval)

# 绘制学习曲线
plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

print('lambda\t\t训练误差\t验证误差')
for i in range(lambda_vec.size):
	print(' {}\t{:.6f}\t{:.6f}'.format(float(lambda_vec[i]), float(error_train[i]), float(error_val[i])))

# 获取最小的交叉验证误差所对应的lambda
p = np.argmin(error_val)
ilambda = lambda_vec[p]

# 进行测试误差的计算
# 进行了规范化所有的样本就都需要规范化, 误差计算不带正则项
error_test, _ = my_lr.computeCostReg(theta, 0, X=X_poly_test, y=ytest)
print('\n最佳lambda: {}\n测试误差: {:.6f}\n'.format(float(ilambda), float(error_test)))






