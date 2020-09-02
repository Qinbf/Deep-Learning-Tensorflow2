
# coding: utf-8

# In[4]:


import tensorflow as tf
import json
 
# 读入json文件
with open('model.json') as m:
    json_config = json.load(m)
    
# 载入json模型结构得到模型model
model = tf.keras.models.model_from_json(json_config)

# summary用于查看模型结构
model.summary()

