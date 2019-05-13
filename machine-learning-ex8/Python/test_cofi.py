"""

            协同过滤测试文件

"""
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from collaborativeFiltering import CollaborativeFiltering as cf
import collaborativeFilteringFunctions as cff

# ~ ## ============== Part 1: 获取用户的评分 ===============
# ~ #

# ~ # 加载电影列表
movieList = cff.load_movie_list()
# 我的评分
my_ratings = np.zeros((movieList.shape[0], 1))
my_ratings[1] = 4
my_ratings[98] = 2
my_ratings[7] = 3
my_ratings[12]= 5
my_ratings[54] = 4
my_ratings[64]= 5
my_ratings[66]= 3
my_ratings[69] = 5
my_ratings[183] = 4
my_ratings[226] = 5
my_ratings[355]= 5
my_ratings[123]= 5
my_ratings[1589]= 5
my_ratings[789]= 5
# ~ print('New user ratings:');
# ~ for i in range(my_ratings.shape[0]):
    # ~ if my_ratings[i] > 0:
        # ~ print('Rated {} for {}'.format(float(my_ratings[i]), movieList[i][0]))


## ================== Part 2: 学习电影特征 ====================
#

print('\nTraining collaborative filtering...')

load_data = sio.loadmat('ex8_movies.mat');
Y = load_data['Y']  # Y is a 1682x943 matrix
R = load_data['R']  # R is a 1682x943 matrix

# 将我的评分加入Y, R
Y = np.hstack((my_ratings, Y))
temp = np.zeros(my_ratings.shape)
temp[np.where(my_ratings!=0)] = 1
R = np.hstack((temp, R))

#  Normalize Ratings 解决没有用户没有进行评分的问题
Ynorm, Ymean = cff.normalize_ratings(Y, R)

# 设置需要训练的特征数目, 以及初始化X Theta
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
# 将X,Theta展开为一个长向量
initial_parameters = np.vstack((X.reshape(X.size, 1),
        Theta.reshape(Theta.size, 1)))

#设置训练参数
ilambda = 10
alpha = 0.005
num_iters = 100

# 训练模型
my_cofi = cf(initial_parameters, Ynorm, R, num_users, num_movies,
        num_features, ilambda)
my_cofi.run(alpha, num_iters)

# 将学习到的X, Theta展开
X = my_cofi.X
Theta = my_cofi.Theta

# 进行特征参数的保存
sio.savemat('parameter.mat', {'X': X, 'Theta': Theta, 'lambda': ilambda,
        'alpha': alpha, 'num_iters': num_iters, 'Ymean': Ymean})
print('Recommender system learning completed.\n')


# ================== Part 8: 电影推荐 ====================
#

# ~ # 加载特征及参数
load_data = sio.loadmat('parameter.mat')
X = load_data['X']
Theta = load_data['Theta']
Ymean = load_data['Ymean']

# 预测
p = X @ Theta.T + Ymean

# 第一列是我自己, 所以这里是为我推荐
my_predictions = p[:, 1]

# 获取预测评分最大的索引
idx = np.argsort(my_predictions)[::-1]

print('Top recommendations for you:')
for i in range(15):
    j = idx[i]
    print('Predicting rating {:.1f} for movie {}'.format(
            my_predictions[j], movieList[j][0]))

print('\nOriginal ratings provided:')
for i in range(my_ratings.shape[0]):
    if my_ratings[i] > 0:
        print('Rated {:.1f} for movie {}'.format(
			float(my_ratings[i]), movieList[i][0]))

