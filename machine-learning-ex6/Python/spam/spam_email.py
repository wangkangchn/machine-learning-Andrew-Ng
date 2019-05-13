"""


                机器学习---支持向量机进行垃圾邮件的分类


"""
import os
import re
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from svmutil import *
import spamFunctions as sf

# ~ ## ==================== Part 1: 对邮件进行预处理 ====================
# ~ #
# ~ print('预处理样本邮件 (emailSample1.txt)')

# ~ # 提取特征
# ~ with open('emailSample1.txt') as f:
    # ~ file_contents = f.read()

# ~ word_indices  = sf.process_email(file_contents)

# ~ # 显示下标
# ~ print('Word Indices: \n')
# ~ print(word_indices)


# ~ ## ==================== Part 2: Feature Extraction ====================
# ~ #
# ~ print('从邮件中提取特征值 (emailSample1.txt)\n');

# ~ with open('emailSample1.txt') as f:
    # ~ file_contents = f.read()
# ~ word_indices  = sf.process_email(file_contents)
# ~ features      = sf.email_features(word_indices)

# ~ # 显示提取结果
# ~ print('Length of feature vector: {}\n'.format(len(features)))
# ~ print('Number of non-zero entries: {}\n'.format(features[features > 0].size))


# ~ ## =========== Part 3: 使用高斯核训练分类器 ========
# ~ #
# ~ print('加载数据...\n')

# ~ # 加载数据集2的数据
# ~ load_data = sio.loadmat('spamTrain.mat')
# ~ X = load_data["X"]
# ~ y = load_data['y']
# ~ train_fn = 'train_data.txt'

# ~ # 数据整理 第一列为标签, 其后为特征
# ~ with open(train_fn, 'w') as f:
    # ~ for label, feature in zip(y, X):
        # ~ # 第一列加入标记
        # ~ msg = '{} '.format(int(label))
        # ~ for i in range(feature.shape[0]):
            # ~ # 后面加入特征
            # ~ msg += '{}:{} '.format(i+1, feature[i])
        # ~ msg += '\n'
        # ~ f.write(msg)

# ~ print('使用高斯核训练模型...\n');
# ~ train_label, train_value = svm_read_problem(train_fn)
# ~ model = svm_train(train_label, train_value, '-c 10 -t 2')

# ~ # 保存训练参数
# ~ svm_save_model('model_spamTrain.mat', model)

# ~ print('预测...\n');
# ~ model = svm_load_model('model_spamTrain.mat')
# ~ svm_predict(train_label, train_value, model)


# ~ ## =================== Part 4: 测试垃圾邮件分类器 ================
# ~ #
# ~ print('加载数据...\n')

# ~ # 加载数据集2的数据
# ~ load_data = sio.loadmat('spamTest.mat')
# ~ Xtest = load_data["Xtest"]
# ~ ytest = load_data['ytest']

# ~ test_fn = 'spamTest.txt'

# ~ # 数据整理 第一列为标签, 其后为特征
# ~ with open(test_fn, 'w') as f:
    # ~ for label, feature in zip(ytest, Xtest):
        # ~ # 第一列加入标记
        # ~ msg = '{} '.format(int(label))
        # ~ for i in range(feature.shape[0]):
            # ~ # 后面加入特征
            # ~ msg += '{}:{} '.format(i+1, feature[i])
        # ~ msg += '\n'
        # ~ f.write(msg)

# ~ test_label, test_value = svm_read_problem(test_fn)
# ~ print('预测...\n');
# ~ model = svm_load_model('model_spamTrain.mat')
# ~ svm_predict(test_label, test_value, model)


## =================== Part 6: Try Your Own Emails =====================
#

filename = 'spamSample2.txt';
# 提取特征
with open(filename) as f:
    file_contents = f.read()

word_indices  = sf.process_email(file_contents)
x     		  = sf.email_features(word_indices)

# 生成支持向量机所需要的数据格式, 因为是预测, 所以标签我们直接是0
y = [0]
# 特征为一个包含字典的列表
x =[{(key+1): float(value) for key,value in zip(range(x.size), x)}] 
# ~ print(x)
model = svm_load_model('model_spamTrain.mat')

p_label, p_acc, p_val = svm_predict(y,x,model)

print('Processed {}\nSpam Classification: {}'.format(filename, int(p_label[0])))
print('(1 indicates spam, 0 indicates not spam)')

# ~ # 负样本目录
# ~ neg_dn = 'spam_2'
# ~ neg_p_dn = 'process_email'
# ~ # 正样本目录
# ~ pos_dn = 'easy_ham'
# ~ pos_p_dn = 'pos_process_email'
# ~ # 训练集
# ~ train_fn = 'train.txt'

# 创建特征词汇表
# ~ print('创建特征词汇表...')
# ~ d, n = sf.creat_freature(neg_dn, neg_p_dn)

# ~ # 排序
# ~ d = sorted(d.items(), key=lambda x:x[1], reverse=True)
# ~ with open('freature_dict_sorted.txt', 'w') as f:
    # ~ for value in d:
        # ~ msg = '{} {}\n'.format(value[1], value[0])
        # ~ f.write(msg)

# ~ # 获取单词列表, 对处理过的文件提取特征
# ~ vocabList = sf.get_vocabList()
# ~ print('特征词汇表创建完毕...')

# 读取处理后的负样本
# ~ print('处理负样本 ...')
# ~ file_names = [files for files in os.walk(neg_p_dn)][0][2]
# ~ sf.extract_features(neg_p_dn, file_names, train_fn, vocabList, 0)
# ~ print('负样本处理完毕 ...')

# 处理正样本
# ~ print('处理正样本 ...')
# ~ sf.process_PosEmail(pos_dn, pos_p_dn)
# ~ # 读取处理后的正样本
# ~ file_names = [files for files in os.walk(pos_p_dn)][0][2]
# ~ sf.extract_features(pos_p_dn, file_names, train_fn, vocabList, 1)
# ~ print('正样本处理完毕 ...')

# 进行训练
# ~ print("进行训练 ...")
# ~ train_label, train_value = svm_read_problem(train_fn)
# ~ model = svm_train(train_label, train_value, '-c 10 -t 2')
# ~ print("训练结束 ...")
# 保存训练参数
# ~ svm_save_model('model.mat', model)

# ~ print('预测...\n');
# ~ model = svm_load_model('model.mat')

# ~ # 创建预测文件
# ~ test_dn = 'test'
# ~ test_p_dn = 'test_process_email'
# ~ test_fn = 'test.txt'

# 处理样本, 因为是进行预测, 所以标签之前是不知道的, 随便设一标签
# ~ sf.process_PosEmail(test_dn, test_p_dn)

# ~ file_names = [files for files in os.walk(test_p_dn)][0][2]
# ~ sf.extract_features(test_p_dn, file_names, test_fn, vocabList, 0)
# ~ test_label, test_value = svm_read_problem(test_fn)

# 进行预测
# ~ p_label, p_acc, p_val = svm_predict(test_label, test_value, model)
# ~ print('p_label: ')
# ~ print(p_label)
# ~ print('p_acc: ')
# ~ print(p_acc)
# ~ print('p_val: ')
# ~ print(p_val)


