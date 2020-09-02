
# coding: utf-8

# In[6]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 载入数据集
mnist = tf.keras.datasets.mnist
# 载入数据，数据载入的时候就已经划分好训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 对训练集和测试集的数据进行归一化处理
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[7]:


import json
import numpy
import requests
# 定义模型签名，可以使用saved_model_cli命令查看
# 定义instances，一次性传入16张图进行预测
data = json.dumps({"signature_name": "serving_default",
                   "instances": x_test[0:16].tolist()})
# 定义headers
headers = {"content-type": "application/json"}
# 定义url，启动tf-serving服务器程序的时候有看到过
# /models/my_model为模型挂载到Docker中的位置
url = 'http://localhost:8501/v1/models/my_model:predict'
# 传输数据进行预测，得到返回结果
json_response = requests.post(url, data=data, headers=headers)
# 对结果进行解析，然后变成array
pre = numpy.array(json.loads(json_response.text)["predictions"])


# In[8]:


print("预测结果为：",np.argmax(pre,axis=1))
print("真实标签为：",y_test[:16])

