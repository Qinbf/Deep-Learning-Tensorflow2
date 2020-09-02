
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout,Conv2D,MaxPool2D,Flatten,AvgPool2D,concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model


# In[2]:


# 类别数
num_classes = 17
# 批次大小
batch_size = 32
# 周期数
epochs = 100
# 图片大小
image_size = 224


# In[3]:


# 训练集数据进行数据增强
train_datagen = ImageDataGenerator(
    rotation_range = 20,     # 随机旋转度数
    width_shift_range = 0.1, # 随机水平平移
    height_shift_range = 0.1,# 随机竖直平移
    rescale = 1/255,         # 数据归一化
    shear_range = 10,       # 随机错切变换
    zoom_range = 0.1,        # 随机放大
    horizontal_flip = True,  # 水平翻转
    brightness_range=(0.7, 1.3), # 亮度变化
    fill_mode = 'nearest',   # 填充方式
) 
# 测试集数据只需要归一化就可以
test_datagen = ImageDataGenerator(
    rescale = 1/255,         # 数据归一化
) 


# In[4]:


# 训练集数据生成器，可以在训练时自动产生数据进行训练
# 从'data/train'获得训练集数据
# 获得数据后会把图片resize为image_size×image_size的大小
# generator每次会产生batch_size个数据
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(image_size,image_size),
    batch_size=batch_size,
    )

# 测试集数据生成器
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(image_size,image_size),
    batch_size=batch_size,
    )


# In[5]:


# 字典的键为17个文件夹的名字，值为对应的分类编号
# train_generator.class_indices


# In[6]:


# 定义Inception结构
def Inception(x,filters):
    tower_1 = Conv2D(filters=filters[0],kernel_size=1,strides=1,padding='same',activation='relu')(x)
    tower_2 = Conv2D(filters=filters[1],kernel_size=1,strides=1,padding='same',activation='relu')(x)
    tower_2 = Conv2D(filters=filters[2],kernel_size=3,strides=1,padding='same',activation='relu')(tower_2)
    tower_3 = Conv2D(filters=filters[3],kernel_size=1,strides=1,padding='same',activation='relu')(x)
    tower_3 = Conv2D(filters=filters[4],kernel_size=5,strides=1,padding='same',activation='relu')(tower_3)
    pooling = MaxPool2D(pool_size=3,strides=1,padding='same')(x)
    pooling = Conv2D(filters=filters[5],kernel_size=1,strides=1,padding='same',activation='relu')(pooling)
    x = concatenate([tower_1,tower_2,tower_3,pooling],axis=3)
    return x
    
# 定义GoogleNet模型
model_input = Input(shape=(image_size,image_size,3))
x = Conv2D(filters=64,kernel_size=7,strides=2,padding='same',activation='relu')(model_input)
x = MaxPool2D(pool_size=3,strides=2,padding='same')(x)
x = Conv2D(filters=64,kernel_size=1,strides=1,padding='same',activation='relu')(x)
x = Conv2D(filters=192,kernel_size=3,strides=1,padding='same',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2,padding='same')(x)
x = Inception(x,[64,96,128,16,32,32])
x = Inception(x,[128,128,192,32,96,64])
x = MaxPool2D(pool_size=3,strides=2,padding='same')(x)
x = Inception(x,[192,96,208,16,48,64])
x = Inception(x,[160,112,224,24,64,64])
x = Inception(x,[128,128,256,24,64,64])
x = Inception(x,[112,144,288,32,64,64])
x = Inception(x,[256,160,320,32,128,128])
x = MaxPool2D(pool_size=3,strides=2,padding='same')(x)
x = Inception(x,[256,160,320,32,128,128])
x = Inception(x,[384,192,384,48,128,128])
x = AvgPool2D(pool_size=7,strides=7,padding='same')(x)
x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(num_classes,activation='softmax')(x)

model = Model(inputs=model_input,outputs=x)

model.summary()


# In[7]:


# 学习率调节函数，逐渐减小学习率
def adjust_learning_rate(epoch):
    # 前40周期
    if epoch<=40:
        lr = 2e-4
    # 前40到80周期
    elif epoch>40 and epoch<=80:
        lr = 2e-5
    # 80到100周期
    else:
        lr = 2e-6
    return lr


# In[8]:


# 定义优化器
adam = Adam(lr=2e-4)

# 定义学习率衰减策略
callbacks = []
callbacks.append(LearningRateScheduler(adjust_learning_rate))

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# Tensorflow2.1版本之前可以使用fit_generator训练模型
# history = model.fit_generator(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=test_generator,validation_steps=len(test_generator))

# Tensorflow2.1版本(包括2.1)之后可以直接使用fit训练模型
history = model.fit(x=train_generator,epochs=epochs,validation_data=test_generator,callbacks=callbacks)


# In[9]:


# 画出训练集准确率曲线图
plt.plot(np.arange(epochs),history.history['accuracy'],c='b',label='train_accuracy')
# 画出验证集准确率曲线图
plt.plot(np.arange(epochs),history.history['val_accuracy'],c='y',label='val_accuracy')
# 图例
plt.legend()
# x坐标描述
plt.xlabel('epochs')
# y坐标描述
plt.ylabel('accuracy')
# 显示图像
plt.show()
# 模型保存
model.save('GoogleNet.h5')

