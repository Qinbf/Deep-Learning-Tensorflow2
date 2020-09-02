
# coding: utf-8

# In[1]:


from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input,Dense,Reshape,Bidirectional,GRU,Lambda
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.callbacks import EarlyStopping,CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import backend as K
from captcha.image import ImageCaptcha  
import matplotlib.pyplot as plt
import numpy as np
import random
import string
from plot_model import plot_model


# In[2]:


# 字符包含所有数字和所有大小写英文字母，一共62个
characters = string.digits + string.ascii_letters
# 类别数+空白字符
num_classes = len(characters)+1
# 批次大小
batch_size = 64
# 训练集批次数
# 训练集大小相当于是64*1000=64000
train_steps = 1000
# 测试集批次数
# 测试集大小相当于是64*100=6400
test_steps = 100
# 周期数
epochs=100
# 图片宽度
width=160
# 图片高度
height=60
# RNN的cell数量
RNN_cell = 128
# 最长验证码
max_len = 6


# In[4]:


# 用于自定义数据生成器
from tensorflow.keras.utils import Sequence

# 这里的Sequence定义其实不算典型，因为一般的数据集数量是有限的，
# 把所有数据训练一次属于训练一个周期，一个周期可以分为n个批次，
# Sequence一般是定义一个训练周期内每个批次的数据如何产生。
# 我们这里的验证码数据集使用captcha模块生产出来的，一边生产一边训练，可以认为数据集是无限的。
class CaptchaSequence(Sequence):
    # __getitem__和__len__是必须定义的两个方法
    def __init__(self, characters, batch_size, steps, n_len=max_len, width=160, height=60,
                 input_len=10, label_len=max_len):
        # 字符集
        self.characters = characters
        # 批次大小
        self.batch_size = batch_size
        # 生成器生成多少个批次的数据
        self.steps = steps
        # 验证码长度随机，3-6位
        self.n_len = np.random.randint(3,7)
        # 验证码图片宽度
        self.width = width
        # 验证码图片高度
        self.height = height
        # 输入长度10，注意这里输入长度指的是RNN模型输出的序列长度，具体要看下面模型搭建部分
        # RNN模型输出序列长度为10表示模型最多可以输入10个字符(包含空白符在内)
        self.input_len = input_len
        # 标签长度
        self.label_len = label_len
        # 字符集长度
        self.num_classes = num_classes
        # 用于产生验证码图片
        self.image = ImageCaptcha(width=self.width, height=self.height)
        # 用于保存最近一个批次验证码字符
        self.captcha_list = []
    
    # 获得index位置的批次数据
    def __getitem__(self, index):
        # 初始化数据用于保存验证码图片
        x = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        # 初始化数据用于保存标签
        y = np.zeros((self.batch_size, self.label_len), dtype=np.int8)
        # 输入长度
        input_len = np.ones(self.batch_size)*self.input_len
        # 标签长度
        label_len = np.ones(self.batch_size)*self.label_len
        # 数据清0
        self.captcha_list = []
        # 生产一个批次数据
        for i in range(self.batch_size):
            # 随机产生验证码
            self.n_len = np.random.randint(3,7)
            # 转字符串
            captcha_text = ''.join([random.choice(self.characters) for j in range(self.n_len)])
            # 保存验证码
            self.captcha_list.append(captcha_text)
            # 生产验证码图片数据并进行归一化处理
            x[i] = np.array(self.image.generate_image(captcha_text)) / 255.0
            for j, ch in enumerate(captcha_text):
                # 设置标签，这里不需要独热编码
                y[i, j] = self.characters.find(ch)
            # 如果验证码长度不是6，则需要设置空白字符
            for k in range(len(captcha_text),self.label_len):
                # 空白字符编号为num_classes-1
                y[i, k] = num_classes-1
        # 返回一个批次的数据和标签
        # 注意这里的标签np.ones(self.batch_size)是没有意义的，只是由于返回的数据必须要有标签
        return [x, y, input_len, label_len], np.ones(self.batch_size)
    
    # 返回批次数量
    def __len__(self):
        return self.steps 


# In[5]:


# 测试生成器
# 一共一个批次，批次大小也是1
data = CaptchaSequence(characters, batch_size=1, steps=1)
for i in range(2):
    # 产生一个批次的数据
    [x, y, _, _], _  = data[0]
    # 显示图片
    plt.imshow(x[0])
    # 验证码字符和对应编号
    plt.title(data.captcha_list[0])
    plt.show()


# In[6]:


# Keras调用Tensorflow中的ctc_batch_cost
# x是模型输出，shape-(?,10,63)
# labels是验证码的标签，shape-(?,max_len)
# input_len是x的长度，shape-(?,1)，x的长度为10
# label_len是labels的的长度，shape-(?,1)，labels的长度为max_len
def ctc_lambda_func(args):
    x, labels, input_len, label_len = args
    # Tensorflow中封装的ctc计算
    return K.ctc_batch_cost(labels, x, input_len, label_len)


# In[7]:


# 载入预训练的resnet50模型，不包含全连接层
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(height,width,3))


# In[8]:


# 设置输入
image_input = Input((height,width,3), name='image_input')
# 使用resnet50进行特征提取
x = resnet50(image_input)
# resnet50计算后得到的数据shape为(?,2,5,2048)
# 10个输入最多对应10个输出，验证码最长为6，理论上只要不出现6个字符都相同的极端情况，长度是够用的。
# 比如极端情况'aaaaaa'，'-'表示空白符，模型输出'a-a-a-a-a-a'至少需要11的长度。
# 不过长度不够可能会影响对连续重复字符的判断效果，比如'aaaa'可能会被识别为'aaa'
# 如果要增加输入长度，可以通过增大输入图片的大小或修改网络结构的方式实现
# 这里Reshape的作用是将卷积输出的4维数据转化为RNN输入所要求的3维数据，2*5=10表示序列长度
x = Reshape((10,2048))(x)
# Bidirectional为双向RNN，可以把RNN/LSTM/GRU传入Bidirectional中
# GRU中的return_sequences=True表示返回所有序列的结果
# 比如在本程序中return_sequences=True返回的结果shape为(?,10,256)
# GRU中的return_sequences=False表示只返回序列last output的结果
# 比如在本程序中return_sequences=False返回的结果shape为(?,256)
x = Bidirectional(GRU(RNN_cell, return_sequences=True))(x)
x = Bidirectional(GRU(RNN_cell, return_sequences=True))(x)
x = Dense(num_classes, activation='softmax')(x)
# 定义模型
model = Model(image_input, x)


# In[9]:


# 定义标签输入
labels = Input(shape=(max_len), name='max_len')
# 输入长度
input_len = Input(shape=(1), name='input_len')
# 标签长度
label_len = Input(shape=(1), name='label_len')
# Lambda的作用是可以将自定义的函数封装到网络中，用于自定义的一些数据计算处理
ctc_out = Lambda(ctc_lambda_func, name='ctc')([x, labels, input_len, label_len])
# 定义模型
ctc_model = Model(inputs=[image_input, labels, input_len, label_len], outputs=ctc_out)
# 画图
plot_model(ctc_model,style=0,show_layer_names=True)


# In[10]:


from tensorflow.keras.callbacks import Callback

# 编号转成字符串
def labels_to_text(labels):
    ret = []
    for l in labels:
        # -1是空白符
        if l == -1:
            ret.append('')
        else:
            ret.append(characters[l])
    return "".join(ret)

# 把一个批次的编号转为字符串
def decode_batch(labels):
    ret = []
    for label in labels:
        ret.append(labels_to_text(label))
    return np.array(ret)

# 自定义Callback
class Evaluate(Callback):
    def __init__(self):
        pass
    # 自定义准确率计算
    def accuracy(self, model, batch_size=batch_size, steps=test_steps):
        # 准确率统计
        batch_acc = 0
        # 产生测试数据
        valid_data = CaptchaSequence(characters, batch_size, steps)
        for [X_test, y_test, _, _], _ in valid_data:
            # 特别要注意，K.ctc_decode会把每个批次的输出统一为相同的长度
            # 如果长短不一，则在短输出后面填充-1，使其每个批次的输出长度对齐
            # 这里可以先将y_test的空白符标签变成-1
            for i,label in enumerate(y_test):
                for j,l in enumerate(label):
                    if l == num_classes-1:
                        y_test[i,j] = -1
            # 将一个批次的标签数据转为字符串形式
            y_test = decode_batch(y_test)
            # 得到预测结果
            y_pred = model.predict(X_test)
            # shape[0]为batch_size，shape[1]为max_len
            shape = y_pred.shape
            # ctc_decode默认使用贪心算法计算出ctc的预测结果
            # get_value获得ctc_decode的数值返回numpy array格式的数据
            out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(shape[0])*shape[1])[0][0])
            # 将一个批次的预测数据转为字符串形式
            out = decode_batch(out)
            # 对比一个批次的标签和预测数据，计算准确率
            batch_acc += (y_test == out).mean()
        # 返回准确率
        return batch_acc / steps
    
    # 顾名思义，在一个训练周期的末尾会自动调用这个方法
    # 这里的epoch是当前训练的周期数
    # logs是一个字典用来记录一些模型训练的信息
    def on_epoch_end(self, epoch, logs):
        # 计算准确率
        acc = self.accuracy(model)
        # 记录val_acc
        logs['val_acc'] = acc
        # 打印
        print(f'\nacc: {acc*100:.4f}')
    
    # 除了on_epoch_end以外，自定义Callback还可以定义很多方法，比如：
    # def on_epoch_begin(self, epoch, logs=None):
    # def on_batch_begin(self, batch, logs=None):
    # def on_batch_end(self, batch, logs=None):
    # 等等，有兴趣的同学可以看tensorflow源码进一步研究。


# In[11]:


# loss的计算是在K.ctc_batch_cost中实现的，所以这里定义了一个假的loss，没什么意义，也没有作用，但是必须要定义
ctc_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=SGD(lr=1e-2,momentum=0.9))

# 监控指标统一使用val_acc
# 可以使用EarlyStopping来让模型停止，连续6个周期val_acc没有上升就结束训练
# CSVLogger保存训练数据
# ModelCheckpoint保存所有训练周期中val_acc最高的模型
# ReduceLROnPlateau学习率调整策略，连续3个周期val_acc没有上升当前学习率乘以0.1
callbacks = [Evaluate(),
             EarlyStopping(monitor='val_acc', patience=6, verbose=1),
             CSVLogger('Captcha_ctc.csv'), 
             ModelCheckpoint('Best_Captcha_ctc.h5', monitor='val_acc', save_best_only=True),
             ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, verbose=1)
             ]


# In[12]:


# 训练模型
ctc_model.fit(x=CaptchaSequence(characters, batch_size=batch_size, steps=train_steps),
              epochs=epochs,
              validation_data=CaptchaSequence(characters, batch_size=batch_size, steps=test_steps),
              callbacks=callbacks)

