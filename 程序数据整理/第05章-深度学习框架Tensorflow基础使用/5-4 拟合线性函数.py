
# coding: utf-8

# In[9]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD

# 使用numpy生成100个从0-1的随机点，作为x
x_data = np.random.rand(100)
# 生成一些随机扰动
noise = np.random.normal(0,0.01,x_data.shape)
# 构建目标值，符合线性分布
y_data = x_data*0.1 + 0.2 + noise

# 画散点图
plt.scatter(x_data, y_data)
plt.show()


# In[10]:


# 构建一个顺序模型
# 顺序模型为keras中的基本模型结构，就像汉堡一样一层一层叠加网络
model = tf.keras.Sequential()
# Dense为全连接层
# 在模型中添加一个全连接层
# units为输出神经元个数，input_dim为输入神经元个数
model.add(tf.keras.layers.Dense(units=1,input_dim=1))
# 设置模型的优化器和代价函数，学习率为0.03
# sgd:Stochastic gradient descent，随机梯度下降法
# mse:Mean Squared Error，均方误差
model.compile(optimizer=SGD(0.03),loss='mse')

# 训练2001个批次
for step in range(2001):
    # 训练一个批次数据，返回cost值
    cost = model.train_on_batch(x_data,y_data)
    # 每500个batch打印一次cost值
    if step % 500 == 0:
        print('cost:',cost)

# 使用predict对数据进行预测，得到预测值y_pred
y_pred = model.predict(x_data)

# 显示随机点
plt.scatter(x_data,y_data)
# 显示预测结果
plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()

