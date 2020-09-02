
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import librosa
from tensorflow.keras.models import load_model
import glob
import os
from tqdm import tqdm
import numpy as np
import sklearn


# In[2]:


# 测试文件存放路径
audio_dir = 'audio_test/'
# 载入模型
model = load_model('audio_model/cnn_0.8943.h5')


# In[3]:


# 获取文件mfcc特征和对应标签
def extract_features(audio_files):
    # 用于保存mfcc特征
    audio_features = []
    # 用于保存标签
    audio_labels = []
    # 由于特征提取需要时间比较长，可以加上tqdm实时查看进度
    for audio in tqdm(audio_files):
        # 读入音频文件
        # 由于音频文件原始采样率高低不一，这里我们把采样率固定为22050
        signal,sample_rate = librosa.load(audio,sr=22050)
        # 由于音频长度长短不一，基本上都在4秒左右，所以我们把所有音频数据的长度都固定为4秒
        # 采样率22050，时长为4秒，所以信号数量为22050*4=88200
        # 小于88200填充
        if len(signal)<88200:
            # 给signal信号前面填充0个数据，后面填充88200-len(signal)个数据，填充值为0
            signal = np.pad(signal,(0,88200-len(signal)),'constant',constant_values=(0))
        # 大于88200，只取前面88200个数据
        else:
            signal = signal[:88200]
        # 获取音频mfcc特征，然后对数据进行转置
        # 原始mfcc数据shape为(mfcc特征数，帧数)->(帧数，mfcc特征数)
        # 相当于把序列长度的维度放前面，特征数的维度放后面
        mfcc = np.transpose(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40), [1,0])
        # 数据标准化
        mfcc = sklearn.preprocessing.scale(mfcc, axis=0)
        # 保存mfcc特征
        audio_features.append(mfcc.tolist()) 
        # 获取label
        # 获取文件名第2个数字，第2个数字为标签
        label = audio.split('/')[-1].split('-')[1]
        # 保存标签
        audio_labels.append(int(label)) 
    return np.array(audio_features), np.array(audio_labels)


# In[4]:


# 获取所有wav文件
audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
print('文件数量：',len(audio_files))


# In[5]:


# 获取文件mfcc特征和对应标签
audio_features,audio_labels = extract_features(audio_files)


# In[6]:


# 把测试数据当作一个批次进行预测
preds = model.predict_on_batch(audio_features)
# 计算概率最大的类别
preds = np.argmax(preds, axis=1)
print('真实标签为：',audio_labels)
print('预测结果为：',preds)

