
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout,Conv2D,MaxPool2D,Flatten,GlobalAvgPool2D,concatenate,BatchNormalization,Activation,Add,ZeroPadding2D,Lambda
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
train_generator.class_indices


# In[6]:


# 定义分组卷积
# g_channels 每组的通道数
# groups 多少组
def grouped_convolution_block(init_x, strides, groups, g_channels):
    group_list = []
    # 分组进行卷积
    for c in range(groups):
        # 分组取出数据
        x = Lambda(lambda x: x[:, :, :, c*g_channels:(c+1)*g_channels])(init_x)
        # 分组进行卷积
        x = Conv2D(filters=g_channels,kernel_size=3,strides=strides,padding='same',use_bias=False)(x)
        # 存入list
        group_list.append(x)
    # 合并list中的数据
    group_merge = concatenate(group_list, axis=3)
    x = BatchNormalization(epsilon=1.001e-5)(group_merge)
    x = Activation('relu')(x)
    return x

# 定义残差单元
def block(x, filters, strides=1, groups=32, conv_shortcut=True): 
    # projection shortcut
    if conv_shortcut == True:
        shortcut = Conv2D(filters*2,kernel_size=1,strides=strides,padding='same')(x)
        # epsilon为BN公式中防止分母为零的值
        shortcut = BatchNormalization(epsilon=1.001e-5)(shortcut)
    else:
        # identity_shortcut
        shortcut = x
    # 3个卷积层
    x = Conv2D(filters=filters,kernel_size=1,strides=1,padding='same')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    # 计算每组的通道数
    g_channels = int(filters / groups) 
    # 进行分组卷积
    x = grouped_convolution_block(x, strides, groups, g_channels)
 
    x = Conv2D(filters=filters*2,kernel_size=1,strides=1,padding='same')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# 堆叠残差单元
def stack(x, filters, blocks, strides, groups=32):
    x = block(x, filters, strides=strides, groups=groups)
    for i in range(blocks):
        x = block(x, filters, groups=groups, conv_shortcut=False)
    return x
    
# 定义ResNeXt50
inputs = Input(shape=(image_size,image_size,3))
# 填充3圈0，填充后图像从224×224变成230×230
x = ZeroPadding2D((3, 3))(inputs)
x= Conv2D(filters=64,kernel_size=7,strides=2,padding='valid')(x)
x = BatchNormalization(epsilon=1.001e-5)(x)
x = Activation('relu')(x)
# 填充1圈0
x = ZeroPadding2D((1, 1))(x)
x = MaxPool2D(pool_size=3,strides=2,padding='valid')(x)
# 堆叠残差结构
# blocks表示堆叠数量
x = stack(x, filters=128, blocks=2, strides=1)
x = stack(x, filters=256, blocks=3, strides=2)
x = stack(x, filters=512, blocks=5, strides=2)
x = stack(x, filters=1024, blocks=2, strides=2)
# 根据特征图大小进行平均池化，池化后得到2维数据
x = GlobalAvgPool2D()(x)
x = Dense(num_classes, activation='softmax')(x)
# 定义模型
model = Model(inputs=inputs,outputs=x)

# 电脑配置不好的话不要运行summary或者plot_model
# model.summary()


# In[7]:


# 学习率调节函数，逐渐减小学习率
def adjust_learning_rate(epoch):
    # 前40周期
    if epoch<=40:
        lr = 1e-4
    # 前40到80周期
    elif epoch>40 and epoch<=80:
        lr = 1e-5
    # 80到100周期
    else:
        lr = 1e-6
    return lr


# In[8]:


# 定义优化器
adam = Adam(lr=1e-4)

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
model.save('ResNeXt50.h5')

