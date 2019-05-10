"""
			神经网络---前向传播
"""

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import one_vs_all_functions as ovaf

#Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10
						 # (note that we have mapped "0" to label 10)

# =========== Part 1: Loading and Visualizing Data =============
print('Loading and Visualizing Data ...\n')

load_fn = 'ex3data1.mat'            #读入mat文件返回的是一个字典
load_data = loadmat(load_fn)

#获取训练集与标记
X = load_data['X']
y = load_data['y']
#训练集大小
m = X.shape[0]

# 随机获取样本的100行, 进行可视化
rand_indices = np.arange(m)
#shuffle返回None 就地打乱
np.random.shuffle(rand_indices)
sel = X[rand_indices[0:100], :]

ovaf.display_data(sel)

# ================ Part 2: Loading Pameters ================
print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
load_theta = loadmat('ex3weights.mat')

theta_1 = load_theta['Theta1']
theta_2 = load_theta['Theta2']

# ~ for key, value in load_theta.items():
	# ~ print('key: {} \n value: {}\n\n'.format(key, value))

# ================= Part 3: Implement Predict =================

pred = ovaf.predict_nn(theta_1, theta_2, X)
print(pred[pred == y])
print('\nTraining Set Accuracy: {}%\n'.format(pred[pred == y].shape[0]/m * 100))

# ~ %  To give you an idea of the network's output, you can also run
# ~ %  through the examples one at the a time to see what it is predicting.

# ~ %  Randomly permute examples
rp = np.arange(m).reshape(m,1)
#shuffle返回None 就地打乱
np.random.shuffle(rp)

for i in range(0, m):
	# Display
	print('\nDisplaying Example Image')
	ovaf.display_data(X[rp[i], :])

	pred = ovaf.predict_nn(theta_1, theta_2, X[rp[i],:])
	print('Neural Network Prediction: {} (digit {})\n'.format(int(pred), 
					int(10 if np.mod(pred, 10) == 0 else np.mod(pred, 10))))
					
	s = input('Paused - press enter to continue, q to exit:');
	if s == 'q':
	  break

