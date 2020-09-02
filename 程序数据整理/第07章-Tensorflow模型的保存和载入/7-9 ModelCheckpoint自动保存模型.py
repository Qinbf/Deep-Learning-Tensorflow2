
# coding: utf-8

# In[38]:


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger


# In[40]:


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
adam = Adam(0.003)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 模型保存位置
output_model = 'ModelCheckpoint/'
# log保存位置
output_log = 'log/'

# ModelCheckpoint用于自动保存模型
# filepath可以设置模型保存位置以及模型信息，epoch表示训练周期数，val_accuracy表示验证集准确值
# monitor可选{'val_accuracy','val_loss','accuracy','loss'},一般'val_accuracy'用得比较多
# verbose=1表示保存模型的时候打印信息
# save_best_only=True表示只保存>best_val_accuracy的模型
# CSVLogger也是callbacks，用于生成模型训练的log
callbacks = [
    ModelCheckpoint(filepath=output_model+'{epoch:02d}-{val_accuracy:.4f}.h5',
                    monitor='val_accuracy',
                    verbose=1,
                    save_best_only=True),
    CSVLogger(output_log + 'log.csv')
]

# 传入训练集数据和标签训练模型
model.fit(x_train, y_train,
          epochs=6, batch_size=32,
          validation_data=(x_test,y_test),
          callbacks=callbacks)

