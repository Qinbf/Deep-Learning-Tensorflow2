
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


# In[3]:


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

# 定义模型输入
inputs = Input(shape=(28,28,1))
x = Conv2D(filters=32,kernel_size=5,strides=1,padding='same',activation='relu')(inputs)
x = MaxPool2D(pool_size=2,strides=2,padding='same')(x)
x = Conv2D(64,5,strides=1,padding='same',activation='relu')(x)
x = MaxPool2D(pool_size=2,strides=2,padding='same')(x)
x = Flatten()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10,activation='softmax')(x)
# 定义模型
model = Model(inputs,x)

# 定义优化器
adam = Adam(lr=1e-4)
# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
# 训练模型
model.fit(x_train,y_train,batch_size=64,epochs=2,validation_data=(x_test, y_test))

