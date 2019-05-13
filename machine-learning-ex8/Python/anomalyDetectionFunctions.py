"""

                异常检查常用函数
                
        estimate_gaussian	-	进行均值以及方差的估计
		multivariate_gaussian-	计算高斯分布概率
		visualize_fit		-	绘制高斯拟合的结果
		select_threshold		-	选择阈值
		
"""
import numpy as np
from matplotlib import pyplot as plt

def estimate_gaussian(X):
    """ 
        估计均值mu和方差sigma2 
        返回列向量
    """
    # 获取数据集大小以及特征值数量
    m, n = X.shape

    # 计算均值
    mu = np.mean(X, 0)
    mu = mu.reshape(mu.size, 1)
    # 计算方差 使用1/m
    sigma2 = np.var(X, axis=0)
    sigma2 = sigma2.reshape(sigma2.size, 1)
    return mu, sigma2

def multivariate_gaussian(X, mu, Sigma2):    # python 变量仅仅是一个标签
    """ 计算多元高斯概率密度值 """
    # 获取特征值数量
    k = mu.shape[0]
    
    # 如果Sigma2是矩阵, 则作为协方差矩阵处理 
    # 如果Sigma2是向量, 则将sigma变为对角矩阵
    # 因为普通的模型中使用的sigma就是协方差的主对角线, 其余元素为0
    if (Sigma2.shape[0] == 1) or (Sigma2.shape[1] == 1):
        # 当sigma2为向量的时候创建对角矩阵
        # Sigma2转换为一维向量后再进行对角矩阵的创建
        Sigma2 = Sigma2.reshape(Sigma2.size)
        Sigma2 = np.diag(Sigma2)
        
    # 计算多元高斯模型
    p = (np.power((2 * np.pi), (- k / 2)) * np.power(np.linalg.det(Sigma2), -0.5) 
            * np.exp(np.sum(-0.5 * (X - mu.T) @ np.linalg.inv(Sigma2) * (X - mu.T), 1)))
    return p
    
def visualize_fit(X, mu, sigma2):
    """ 可视化高斯模型结果 """
    x = np.arange(0, 35.5, 0.5)     # 起:止:步长
    
    # 生成网格数据
    X1, X2 = np.meshgrid(x, x)
    Z = multivariate_gaussian(np.hstack((X1.reshape(X1.size, 1), X2.reshape(X2.size, 1))), 
            mu, sigma2)
    Z = Z.reshape(X1.shape)
    
    # 数据点
    plt.plot(X[:, 0], X[:, 1],'bx')
 
    # 绘制指定数值的等高线
    lines = [10**i for i in range(-20,0,3)]
    # 绘制拟合的等高线图
    plt.contour(X1, X2, Z, lines, cmap='summer')

def select_threshold(yval, pval):
    """ 选择阈值epsilon, 返回epsilon以及对应的F1分 """
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    
    # 临时比较数组
    cvPredictions = np.zeros((pval.shape[0], 1), dtype=np.int32)
    yval_temp = np.zeros((pval.shape[0], 1), dtype=np.int32)
    pred_temp = np.zeros((pval.shape[0], 1), dtype=np.int32)
    # 选择步长
    stepsize = (np.max(pval) - np.min(pval)) / 1000
    for epsilon in np.arange(np.min(pval), np.max(pval)+stepsize, stepsize):
        # 选择阈值epsilon, 计算查准率以及召回率, 计算F分
        # 先使用pval进行预测, 得到预测的数据后使用yval来计算tp, fp, fn
        
        # 依据当前 epsilon 进行预测, 小于epsilon为异常点标记为1
        cvPredictions[np.where(pval < epsilon)[0]] = 1
        cvPredictions[np.where(pval >= epsilon)[0]] = 0

        # 计算tp, 预测以及实际值均为1的个数, 需要对整个数组进行比较
        tp = np.sum(cvPredictions & yval)
        
        # 计算fp, 预测为1实际为0的个数 实际标签1, 0反置
        # 将yval值进行替换1->0 0->1
        yval_temp[np.where(yval == 1)[0]] = 0
        yval_temp[np.where(yval == 0)[0]] = 1
        fp = np.sum(cvPredictions & yval_temp)
        
        # 计算fn, 预测为0实际为1的个数	预测标签1, 0反置
        pred_temp[np.where(cvPredictions == 1)[0]] = 0
        pred_temp[np.where(cvPredictions == 0)[0]] = 1
        fn = np.sum(pred_temp & yval)
        
        # 计算查准率
        prec = tp / (tp + fp)
        
        # 计算召回率
        rec = tp / (tp + fn)
        
        # 计算F分
        F1 = 2 * prec * rec / (prec + rec)
    
        # 获取最佳F1
        if F1 > bestF1:
           bestF1 = F1
           bestEpsilon = epsilon
           
    return bestEpsilon, bestF1
    
