"""
        邮件分类所需要的函数
"""

import re
import os
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import nltk.stem
from svmutil import *

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files              # 当前路径下所有非目录子文件

def get_vocabList(file_name='vocab.txt'):
    # 读取单词
    data = np.loadtxt(file_name, dtype=str)
    vocabList = data[:, 1]
    vocabList = vocabList.reshape(vocabList.size, 1)
    # ~ print(vocabList)
    return vocabList

def simplify_email (email_contents):
    """ 简化邮件的内容 """
    # 去除标题行
    email_contents = re.sub('From .+\n', ' ', email_contents)
    email_contents = re.sub('Return-Path: .+\n', ' ', email_contents)
    email_contents = re.sub('Delivery-Date: .+\n', ' ', email_contents)
    email_contents = re.sub('Received: .+\n', ' ', email_contents)
    email_contents = re.sub('To: .+\n', ' ', email_contents)
    email_contents = re.sub('Message-Id: .+\n', ' ', email_contents)
    email_contents = re.sub('Subject: .+\n', ' ', email_contents)
    email_contents = re.sub('Date: .+\n', ' ', email_contents)
    email_contents = re.sub('MIME-Version: .+\n', ' ', email_contents)
    # hdrstart = strfind(email_contents, ([char(10) char(10)]));
    # email_contents = email_contents(hdrstart(1):end);
    # ~ print("原始emial: \n", email_contents)
    # 全部变为小写字母
    email_contents = email_contents.lower()

    # 将 换行 替换为空格
    email_contents = re.sub('\n', ' ', email_contents)

    # 将 <> 替换为空格    剥去所有的HTML格式
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # 将数字字符替换为number
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # 将URL地址替换为 httpaddr
    email_contents = re.sub('(http|https)://[^\s]*', ' httpaddr', email_contents)

    # 将电子邮箱地址替换为 emailaddr
    email_contents = re.sub('[^\s]+@[^\s]+', ' emailaddr', email_contents)

    # 将美元符号替换为 dollar
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # 去掉标点符号
    email_contents = re.sub('[@$/#.-:&*+=[]?!(){},\'">_<;%]', ' ', email_contents)

    # 删除任何非字母数字字符
    email_contents = re.sub('[^a-zA-Z0-9 ]', '', email_contents)
    # ~ print("处理后emial: \n", email_contents)
    return email_contents

def process_email(email_contents):
    # 加载垃圾词汇表
    vocabList = get_vocabList()

    # 初始化返回值
    word_indices = []

    email_contents = simplify_email (email_contents)
    # ========================== 标记邮件 ===========================

    # 词干提取器
    my_stem = nltk.stem.SnowballStemmer('english')

    # 提取所以的单词进行处理
    for word in re.finditer('\w+', email_contents):
        # 提取词干
        word = word.group()     # 提取单词
        word = my_stem.stem(word)

        # 跳过长度太短的单词
        if len(word) < 1:
           continue

        # 遍历vocabList 进行每一个单词的比较, 获取下标
        for i in range(vocabList.size):
            if word == vocabList[i]:
                # 存在比较成功且进行下标的保存
                word_indices.append(i)
                # 结束循环
                break
            else:
                # 比较失败, 进行下一次循环
                continue
    return word_indices

def email_features(word_indices, n=1899):
    # 词汇表的数量
    x = np.zeros((n, 1))
    for i in range(len(word_indices)):
        # 保存在word_indices中的就是在电子邮件中存在的特征, 所以直接将word_indices
        # 中的下标处置为1即可
        x[word_indices[i]] = 1
    return x

def creat_freature (dn, p_dn):
    """ 读取垃圾邮件找到预处理后出现频率最大的单词 """
    # 词汇字典
    freature_dict = {}
    
    # 词干提取器
    my_stem = nltk.stem.SnowballStemmer('english')

    # 垃圾邮件所在的目录
    dn = dn
    # 处理后的邮件所在的目录
    p_dn = p_dn
    if not os.path.isdir(p_dn):
        os.mkdir(p_dn)
    
    # 获得邮件目录下的所有邮件文件名 返回一个包含元组的列表 
    file_names = [files for files in os.walk(dn)]
    # 去掉根目录以及子目录
    file_names = file_names[0][2]
    for filename in file_names:
        # 读取文件内容
        try:
            with open(dn+'\\'+filename, encoding='utf-8') as f:
                email_contents = f.read()
        except:
            continue
        # 处理邮件内容
        email_contents = simplify_email (email_contents)
        
        #将处理后的邮件保存到文件中
        with open(p_dn+'\\'+filename+'.txt', 'w') as f:
            f.write(email_contents)
        
        
        # 提取所有的单词进行处理
        for word in re.finditer('\w+', email_contents):
            # 提取词干
            word = word.group()     # 提取单词
            word = my_stem.stem(word)

            # 跳过长度太短的单词
            if len(word) < 1:
               continue

            # 记录单词出现的次数, 字典中没有时使用0做默认值
            freature_dict.setdefault(word, 0)
            freature_dict[word] += 1

    with open('freature_dict.txt', 'w') as f:
        for key, value in freature_dict.items():
            msg = '{} {}\n'.format(value, key)
            f.write(msg)
            
    return freature_dict, len(freature_dict)

def process_PosEmail(dn, p_dn):
    """ 对正样本进行处理 """
    # 词干提取器
    my_stem = nltk.stem.SnowballStemmer('english')

    # 垃圾邮件所在的目录
    dn = dn
    # 处理后的邮件所在的目录
    p_dn = p_dn
    if not os.path.isdir(p_dn):
        os.mkdir(p_dn)
    
    # 获得邮件目录下的所有邮件文件名 返回一个包含元组的列表 
    file_names = [files for files in os.walk(dn)]
    # 去掉根目录以及子目录
    file_names = file_names[0][2]
    for filename in file_names:
        # 读取文件内容
        try:
            with open(dn+'\\'+filename, encoding='utf-8') as f:
                email_contents = f.read()
        except:
            continue
        # 处理邮件内容
        email_contents = simplify_email (email_contents)
        
        #将处理后的邮件保存到文件中
        with open(p_dn+'\\'+filename+'.txt', 'w') as f:
            f.write(email_contents)
    
def extract_features(p_dn, file_names, train_fn, vocabList, label):
    print('特征抽取...')
    
    for file_name in file_names:       
        print('{} ...'.format(file_name))
        
        word_indices = []
        
        with open(p_dn+'\\'+file_name) as f:
            email_contents = f.read()
        
        # 遍历vocabList 进行每一个单词的比较, 获取下标
        for word in re.finditer('\w+', email_contents):
            word = word.group()     # 提取单词
            
            for i in range(vocabList.size):
                if word == vocabList[i]:
                    # 存在比较成功且进行下标的保存
                    word_indices.append(i)
                    # 结束循环
                    break
                else:
                    # 比较失败, 进行下一次循环
                    continue
    
        # 特征提取
        X = email_features(word_indices, vocabList.size)
        # 数据整理 第一列为标签, 其后为特征
        with open(train_fn, 'a') as f:
            # 第一列为标签
            msg = '{} '.format(label) 
            for i in range(X.size):
                # 后面加入特征
                msg += '{}:{} '.format(i+1, int(X[i]))
            msg += '\n'
            f.write(msg)
