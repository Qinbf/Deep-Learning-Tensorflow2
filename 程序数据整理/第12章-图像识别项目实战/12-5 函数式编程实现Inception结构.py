
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input,Conv2D,MaxPool2D,concatenate
from tensorflow.keras.models import Model


# In[2]:


# 定义模型输入
inputs = Input(shape=(28,28,192))
# 注意函数式模型的特点，Conv2D后面的(inputs)表示把inputs信号输入到Conv2D中计算
tower_1 = Conv2D(filters=64,kernel_size=(1,1),strides=(1,1),padding='same',activation='relu')(inputs)
# 注意函数式模型的特点，Conv2D后面的(inputs)表示把inputs信号输入到Conv2D中计算
tower_2 = Conv2D(filters=96,kernel_size=(1,1),strides=(1,1),padding='same',activation='relu')(inputs)
# 注意函数式模型的特点，Conv2D后面的(tower_2)表示把tower_2信号输入到Conv2D中计算
tower_2 = Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(tower_2)
# 注意函数式模型的特点，Conv2D后面的(inputs)表示把inputs信号输入到Conv2D中计算
tower_3 = Conv2D(filters=16,kernel_size=(1,1),strides=(1,1),padding='same',activation='relu')(inputs)
# 注意函数式模型的特点，Conv2D后面的(tower_3)表示把tower_3信号输入到Conv2D中计算
tower_3 = Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu')(tower_3)
# 注意函数式模型的特点，MaxPool2D后面的(inputs)表示把inputs信号输入到MaxPool2D中计算
pooling = MaxPool2D(pool_size=(3, 3),strides=(1, 1),padding='same')(inputs)
# 注意函数式模型的特点，Conv2D后面的(pooling)表示把pooling信号输入到Conv2D中计算
pooling = Conv2D(filters=32,kernel_size=(1,1),strides=(1,1),padding='same',activation='relu')(pooling)
# concatenate合并4个信号，axis=3表示根据channel进行合并，得到模型的输出
outputs = concatenate([tower_1,tower_2,tower_3,pooling],axis=3)
# 定义模型，设置输入和输出信号
model = Model(inputs=inputs, outputs=outputs)
# 查看模型概要
model.summary()

