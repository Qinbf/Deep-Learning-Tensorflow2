
# coding: utf-8

# In[1]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout

# 模型定义
model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=200,activation='tanh'),
        Dropout(0.4),
        Dense(units=100,activation='tanh'),
        Dropout(0.4),
        Dense(units=10,activation='softmax')
        ])

# 保存模型结构
config = model.get_config()
print(config)


# In[2]:


# 保存模型结构
json_config = model.to_json()
print(json_config)


# In[3]:


import json
# 保存json模型结构文件
with open('model.json','w') as m:
    json.dump(json_config,m)


# In[14]:


import numpy as np
a = np.array(model.get_weights())
for i in a:
    print(i.shape)


# In[ ]:


for i in 

