"""
        
                协同过滤算法


"""
import numpy as np

def cofi_costfunc(params, Y, R, num_users, num_movies, num_features, ilambda=0):
    """ 代价函数 params为X和Theta组成的长向量"""
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)
    
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    
    # 先进行计算所有值的预测误差, 再与R元素相乘提取所需要的元素
    J = (0.5 * np.sum(np.sum(np.power(R * (X @ Theta.T - Y), 2))) 
            + ilambda / 2 * np.sum(np.sum(np.power(X, 2))) 
            + ilambda / 2 * np.sum(np.sum(np.power(Theta, 2))))
    
    # 计算梯度
    X_grad = (R * (X @ Theta.T - Y)) @ Theta + ilambda * X          # N_m*N_u   N_u*n
    Theta_grad = (R * (X @ Theta.T - Y)).T @ X + ilambda * Theta    # N_u*N_m   N_m*n
    
    # 将梯度展开为一个长向量
    grad = np.vstack((X_grad.reshape(X_grad.size, 1), 
                    Theta_grad.reshape(Theta_grad.size, 1)))
    
    return J, grad
    
def compute_numerical_gradient(params, Y, R, num_users, num_movies, num_features, ilambda):
    """ 进行近似的数值梯度计算 """         
    numgrad = np.zeros((params.size, 1))
    perturb = np.zeros((params.size, 1))
    e = 1e-4
    # 已经将参数展成了一个长向量, 对每一个参数进行梯度检查, 检查时仅修改该参数, 其余参数不变
    for p in range(params.size):
        # 设置偏移值    
        perturb[p] = e
        loss1, _ = cofi_costfunc(params - perturb, Y, R, num_users, num_movies, num_features, ilambda)
        loss2, _ = cofi_costfunc(params + perturb, Y, R, num_users, num_movies, num_features, ilambda)
        # 计算数值梯度
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0  # 恢复避免扰乱其他的值计算
    return numgrad

def check_cost_function (ilambda=0):
    """ 
        创建小型问题进行梯度检测
    
    """ 
    # 3个特征值, 4部电影, 5位用户
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)
    
    # 创建随机的未评分位置
    Y = X_t @ Theta_t.T
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1
    
    ## 运行梯度检测
    X = np.random.randn(X_t.shape[0], X_t.shape[1])     # 初始化特征
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1]) # 初始化Theta
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta.shape[1]
    
    # 将X和Theta组合起来当做一个参数
    params = np.vstack((X.reshape(X.size, 1), Theta.reshape(Theta.size, 1)))
    # 进行分析梯度计算
    cost, grad = cofi_costfunc(params, Y, R, num_users, num_movies, num_features, ilambda)   
    # 进行数值梯度计算, 
    numgrad = compute_numerical_gradient(params, Y, R, num_users, num_movies, num_features, ilambda)
    
    # 将梯度展开为长向量
    # 显示分析计算的梯度以及近似计算得到的数值梯度
    for num, analyze in zip(numgrad, grad):     # zip可以使多个迭代器一起迭代
        print('{} {}'.format(num, analyze))
    print('\n上面的两列应该非常的相近.\n' 
            '(左-数值梯度, 右-解析梯度)\n')
    
    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in compute_numerical_gradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    
    print('If your cost function implementation is correct, then\n '
         'the relative difference will be small (less than 1e-9). \n'
         'Relative Difference: {}\n'.format(diff))
        
def load_movie_list ():
    """ 加载电影列表 """
    # 加载文件内容
    movieList = []
    with open('movie_ids.txt') as f:
        for line in f:
            movieList.append(line[line.find(' '):-1])
    movieList = np.array(movieList)
    movieList = movieList.reshape(movieList.size, 1)
    return movieList

def normalize_ratings(Y, R):
    """ 均值归一化 """   
    m, n = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(Y.shape)
    
    # 计算每一行的均值
    for i in range(m):
        idx = np.where(R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean
