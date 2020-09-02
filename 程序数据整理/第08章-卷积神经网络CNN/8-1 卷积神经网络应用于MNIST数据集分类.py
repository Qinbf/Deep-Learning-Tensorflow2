
# coding: utf-8

# In[5]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import Adam


# In[6]:


# 载入数据
mnist = tf.keras.datasets.mnist
# 载入数据，数据载入的时候就已经划分好训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 这里要注意，在tensorflow中，在做卷积的时候需要把数据变成4维的格式
# 这4个维度是(数据数量，图片高度，图片宽度，图片通道数)
# 所以这里把数据reshape变成4维数据，黑白图片的通道数是1，彩色图片通道数是3
x_train = x_train.reshape(-1,28,28,1)/255.0
x_test = x_test.reshape(-1,28,28,1)/255.0
# 把训练集和测试集的标签转为独热编码
y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)

# 定义顺序模型
model = Sequential()
# 第一个卷积层
# input_shape 输入数据
# filters 滤波器个数32，生成32张特征图
# kernel_size 卷积窗口大小5*5
# strides 步长1
# padding padding方式 same/valid
# activation 激活函数
model.add(Convolution2D(
    input_shape = (28,28,1),
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    activation = 'relu'
))
# 第一个池化层
# pool_size 池化窗口大小2*2
# strides 步长2
# padding padding方式 same/valid
model.add(MaxPooling2D(
    pool_size = 2,
    strides = 2,
    padding = 'same',
))
# 第二个卷积层
# filters 滤波器个数64，生成64张特征图
# kernel_size 卷积窗口大小5*5
# strides 步长1
# padding padding方式 same/valid
# activation 激活函数
model.add(Convolution2D(64,5,strides=1,padding='same',activation='relu'))
# 第二个池化层
# pool_size 池化窗口大小2*2
# strides 步长2
# padding padding方式 same/valid
model.add(MaxPooling2D(2,2,'same'))
# 把第二个池化层的输出进行数据扁平化
# 相当于把(64,7,7,64)数据->(64,7*7*64)
model.add(Flatten())
# 第一个全连接层
model.add(Dense(1024,activation = 'relu'))
# Dropout
model.add(Dropout(0.5))
# 第二个全连接层
model.add(Dense(10,activation='softmax'))
# 定义优化器
adam = Adam(lr=1e-4)
# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
# 训练模型
model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_test, y_test))


# In[7]:


# 保存模型
model.save('mnist.h5')

