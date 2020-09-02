
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Activation
from tensorflow.keras.optimizers import Adam,RMSprop
import matplotlib.pyplot as plt


# In[2]:


# 下载并载入数据
# 训练集数据(50000, 32, 32, 3)
# 测试集数据(50000, 1)
(x_train,y_train),(x_test,y_test) = cifar10.load_data()


# In[4]:


# 数据归一化
x_train = x_train/255.0
x_test = x_test/255.0
# 换one hot格式
y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)


# In[5]:


# 定义卷积网络
model = Sequential()
model.add(Conv2D(input_shape=(32,32,3), filters=32, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(10,activation = 'softmax'))


# 定义使用了BN的卷积网络
# 两个模型结构完全一致，区别只在于是否使用BN
model_bn = Sequential()
model_bn.add(Conv2D(input_shape=(32,32,3), filters=32, kernel_size=3, strides=1, padding='same'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model_bn.add(Dropout(0.2))

model_bn.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model_bn.add(Dropout(0.3))

model_bn.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same'))
model_bn.add(BatchNormalization())
model_bn.add(Activation('relu'))
model_bn.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model_bn.add(Dropout(0.4))

model_bn.add(Flatten())
model_bn.add(Dense(10,activation = 'softmax'))

# 定义优化器
adam = Adam(lr=1e-4)

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
# 定义优化器，loss function，训练过程中计算准确率
model_bn.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])


# In[6]:


# 训练模型
history = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), shuffle=True)
history_bn = model_bn.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), shuffle=True)


# In[9]:


# 画出没有使用BN的模型验证集准确率
plt.plot(np.arange(100),history.history['val_accuracy'],c='b',label='without_bn')
# 画出使用BN的模型验证集准确率
plt.plot(np.arange(100),history_bn.history['val_accuracy'],c='y',label='bn')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

