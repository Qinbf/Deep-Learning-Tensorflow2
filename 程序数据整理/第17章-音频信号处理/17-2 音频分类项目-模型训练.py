
# coding: utf-8

# In[6]:


import warnings
warnings.filterwarnings("ignore")
import glob
import os
# 需要安装tqdm，用于查看进度条
# pip install tqdm
from tqdm import tqdm 
# pip install librosa
import librosa
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
# pip install plot_model
from plot_model import plot_model


# In[81]:


# 音频文件存放位置
# 在'audio/'文件夹下还有fold1，fold2，fold3这3个文件夹
audio_dir = 'audio/'
# 批次大小
batch_size = 64
# 训练周期
epochs = 500


# In[53]:


# 获取所有wav文件
def get_wav_files(audio_dir):
    # 用于保存音频文件路径
    audio_files = []
    # 循环文件夹
    for sub_file in os.listdir(audio_dir):
        # 得到文件完整路径
        file = os.path.join(audio_dir,sub_file)
        # 如果是文件夹
        if os.path.isdir(file):
            # 得到file文件夹下所有'*.wav'文件
            audio_files += glob.glob(os.path.join(file, '*.wav'))
    return audio_files

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


# In[54]:


# 获取所有wav文件
audio_files = get_wav_files(audio_dir)
print('文件数量：',len(audio_files))


# In[10]:


# 获取文件mfcc特征和对应标签
audio_features,audio_labels = extract_features(audio_files)


# In[12]:


# 切分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(audio_features,audio_labels)


# In[107]:


from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv1D,GlobalMaxPool1D,AlphaDropout,Dense,Input,concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import selu
from tensorflow.keras.callbacks import EarlyStopping,CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
# 定义模型输入
inputs = Input(shape=(x_train.shape[1:]))
# 定义1维卷积，权值初始化使用lecun_normal，主要是为了跟selu搭配
x0 = Conv1D(filters=256, kernel_size=3, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))(inputs)
x0 =GlobalMaxPool1D()(x0)
# 定义1维卷积
x1 = Conv1D(filters=256, kernel_size=4, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))(inputs)
x1 =GlobalMaxPool1D()(x1)
# 定义1维卷积
x2 = Conv1D(filters=256, kernel_size=5, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.0001))(inputs)
x2 =GlobalMaxPool1D()(x2)
# 合并特征
x = concatenate([x0,x1,x2],axis=-1)
# 可以用AlphaDropout保持信号均值和方差不变，AlphaDropout一般跟selu搭配
x = AlphaDropout(0.5)(x)
# 10分类
preds = Dense(10, activation='softmax', kernel_initializer='lecun_normal')(x)
# 定义模型
model = Model(inputs, preds)
# 画结构图
plot_model(model, dpi=200)


# In[108]:


# 定义优化器
# 因为标签没有转独热编码，所以loss用sparse_categorical_crossentropy
model.compile(optimizer=Adam(0.01),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# 监控指标统一使用val_accuracy
# 可以使用EarlyStopping来让模型停止，连续40个周期val_accuracy没有下降就结束训练
# ModelCheckpoint保存所有训练周期中val_accuracy最高的模型
# ReduceLROnPlateau学习率调整策略，连续20个周期val_accuracy没有提升，当前学习率乘以0.1
callbacks = [EarlyStopping(monitor='val_accuracy', patience=40, verbose=1),
             ModelCheckpoint('audio_model/'+'cnn_{val_accuracy:.4f}.h5', monitor='val_accuracy', save_best_only=True),
             ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=20, verbose=1)]

# 模型训练
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=callbacks)

