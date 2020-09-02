
# coding: utf-8

# In[4]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout
# 载入模型参数前需要先把模型定义好
# 模型结构需要与参数匹配
# 或者可以使用tf.keras.models.model_from_json载入模型结构
model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=200,activation='tanh'),
        Dropout(0.4),
        Dense(units=100,activation='tanh'),
        Dropout(0.4),
        Dense(units=10,activation='softmax')
        ])

# 载入模型参数
model.load_weights('my_model/model_weights.h5')

