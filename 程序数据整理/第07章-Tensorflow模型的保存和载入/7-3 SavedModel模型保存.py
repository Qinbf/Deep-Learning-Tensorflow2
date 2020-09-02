
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.optimizers import SGD


# In[2]:


# 载入数据集
mnist = tf.keras.datasets.mnist
# 载入数据，数据载入的时候就已经划分好训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 对训练集和测试集的数据进行归一化处理，有助于提升模型训练速度
x_train, x_test = x_train / 255.0, x_test / 255.0
# 把训练集和测试集的标签转为独热编码
y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)

# 模型定义
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 定义优化器，代价函数
sgd = SGD(0.2)
model.compile(optimizer=sgd,
              loss='mse',
              metrics=['accuracy'])

# 传入训练集数据和标签训练模型
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test,y_test))

# 保存模型为SavedModel格式
model.save('path_to_saved_model')  


# In[3]:


model.to_json()

