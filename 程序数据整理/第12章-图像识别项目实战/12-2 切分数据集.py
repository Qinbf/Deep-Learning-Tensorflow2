
# coding: utf-8

# In[1]:


import os
import random
import shutil
import numpy as np


# In[2]:


# 数据集路径
DATASET_DIR = "new_17_flowers"
# 数据切分后存放路径
NEW_DIR = "data"
# 测试集占比
num_test = 0.2


# In[3]:


# 打乱所有种类数据，并分割训练集和测试集
def shuffle_all_files(dataset_dir, new_dir, num_test):
    # 先删除已有new_dir文件夹
    if not os.path.exists(new_dir):
        pass
    else:
        # 递归删除文件夹
        shutil.rmtree(new_dir)
    # 重新创建new_dir文件夹
    os.makedirs(new_dir)
    # 在new_dir文件夹目录下创建train文件夹
    train_dir = os.path.join(new_dir, 'train')
    os.makedirs(train_dir)
    # 在new_dir文件夹目录下创建test文件夹
    test_dir = os.path.join(new_dir, 'test')
    os.makedirs(test_dir)
    # 原始数据类别列表
    directories = []
    # 新训练集类别列表
    train_directories = [] 
    # 新测试集类别列表
    test_directories = [] 
    # 类别名称列表
    class_names = []
    # 循环所有类别
    for filename in os.listdir(dataset_dir):
        # 原始数据类别路径
        path = os.path.join(dataset_dir, filename)
        # 新训练集类别路径
        train_path = os.path.join(train_dir, filename)
        # 新测试集类别路径
        test_path = os.path.join(test_dir, filename)
        # 判断该路径是否为文件夹
        if os.path.isdir(path):
            # 加入原始数据类别列表
            directories.append(path)
            # 加入新训练集类别列表
            train_directories.append(train_path)
            # 新建类别文件夹
            os.makedirs(train_path)
            # 加入新测试集类别列表
            test_directories.append(test_path)
            # 新建类别文件夹
            os.makedirs(test_path)
            # 加入类别名称列表
            class_names.append(filename)
    print('类别列表：',class_names)
    
    # 循环每个分类的文件夹
    for i in range(len(directories)):
        # 保存原始图片路径
        photo_filenames = []
        # 保存新训练集图片路径
        train_photo_filenames = []
        # 保存新测试集图片路径
        test_photo_filenames = []
        # 得到所有图片的路径
        for filename in os.listdir(directories[i]):
            # 原始图片路径
            path = os.path.join(directories[i], filename)
            # 训练图片路径
            train_path = os.path.join(train_directories[i], filename)
            # 测试集图片路径
            test_path = os.path.join(test_directories[i], filename)
            # 保存图片路径
            photo_filenames.append(path)
            train_photo_filenames.append(train_path)
            test_photo_filenames.append(test_path)
        # list转array
        photo_filenames = np.array(photo_filenames)
        train_photo_filenames = np.array(train_photo_filenames)
        test_photo_filenames = np.array(test_photo_filenames)
        # 打乱索引
        index = [i for i in range(len(photo_filenames))] 
        random.shuffle(index)
        # 对3个list进行相同的打乱，保证在3个list中索引一致
        photo_filenames = photo_filenames[index]
        train_photo_filenames = train_photo_filenames[index]
        test_photo_filenames = test_photo_filenames[index]
        # 计算测试集数据个数
        test_sample_index = int((1-num_test) * float(len(photo_filenames)))
        # 复制测试集图片
        for j in range(test_sample_index, len(photo_filenames)):
            # 复制图片
            shutil.copyfile(photo_filenames[j], test_photo_filenames[j])
        # 复制训练集图片
        for j in range(0, test_sample_index):
            # 复制图片
            shutil.copyfile(photo_filenames[j], train_photo_filenames[j])


# In[4]:


# 打乱并切分数据集
shuffle_all_files(DATASET_DIR, NEW_DIR, num_test)

