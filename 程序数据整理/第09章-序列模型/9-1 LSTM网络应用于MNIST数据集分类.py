
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


# In[8]:


# 载入数据集
mnist = tf.keras.datasets.mnist
# 载入数据，数据载入的时候就已经划分好训练集和测试集
# 训练集数据x_train的数据形状为（60000，28，28）
# 训练集标签y_train的数据形状为（60000）
# 测试集数据x_test的数据形状为（10000，28，28）
# 测试集标签y_test的数据形状为（10000）
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 对训练集和测试集的数据进行归一化处理，有助于提升模型训练速度
x_train, x_test = x_train / 255.0, x_test / 255.0
# 把训练集和测试集的标签转为独热编码
y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)

# 数据大小-一行有28个像素
input_size = 28
# 序列长度-一共有28行
time_steps = 28
# 隐藏层memory block个数
cell_size = 50 

# 创建模型
model = Sequential()

# 循环神经网络的数据输入必须是3维数据
# 数据格式为(数据数量，序列长度，数据大小)
# 载入的mnist数据的格式刚好符合要求
# 注意这里的input_shape设置模型数据输入时不需要设置数据的数量
model.add(LSTM(
    units = cell_size, 
    input_shape = (time_steps,input_size),
))

# 50个memory block输出的50个值跟输出层10个神经元全连接
model.add(Dense(10,activation='softmax'))

# 定义优化器
adam = Adam(lr=1e-3)

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# 训练模型
model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_test,y_test))

