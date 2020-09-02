
# coding: utf-8

# In[5]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout
import numpy as np

# 模型定义
model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=200,activation='tanh'),
        Dropout(0.4),
        Dense(units=100,activation='tanh'),
        Dropout(0.4),
        Dense(units=10,activation='softmax')
        ])

# 保存模型参数
model.save_weights('my_model/model_weights')

# 获取模型参数
weights = model.get_weights()
# 把list转变成array
weights = np.array(weights)

# 循环每一层权值
# enumerate相当于循环计数器，记录当前循环次数
# weights保存的数据可以对照print输出查看
for i,w in enumerate(weights):
    if i%2==0:
        print('{}:w_shape:{}'.format(int(i/2+1),w.shape))
    else:
        print('{}:b_shape:{}'.format(int(i/2+0.5),w.shape))

