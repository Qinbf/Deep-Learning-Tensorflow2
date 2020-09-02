
# coding: utf-8

# In[1]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import json
import os
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# 测试图片存放位置
image_dir = 'flowers_test'


# In[3]:


# 载入标签json文件
file = open('label_flower.json','r',encoding='utf-8')
label = json.load(file)
# 键为分类编号，值为分类名称
print(label)


# In[4]:


# 载入训练好的模型
model = load_model('VGG16.h5')


# In[5]:


def model_predict(file_dir):
    # 读入图片，并resize为224*224大小
    img = load_img(file_dir, target_size=(224, 224))
    # 显示图片
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    # 将图片转化为array
    x = img_to_array(img)
    # 增加1个维度变成4维数据
    # (224, 224, 3)->(1, 224, 224, 3)
    x = np.expand_dims(x, axis=0) 
    # 模型预测结果
    # predict_classes直接返回预测分类结果，比如:[2]
    preds = model.predict_classes(x)
    # label字典中的键为字符串，所以这里需要把preds[0]转为str
    # 根据分类编号查询label中对应的分类名称
    preds = label[str(preds[0])]
    return preds

# 循环测试文件夹
for file in os.listdir(image_dir):
    # 测试图片完整路径
    file_dir = os.path.join(image_dir,file)
    # 打印文件路径
    print(file_dir)
    # 传入文件路径进行预测
    preds = model_predict(file_dir)
    print('predict:',preds)
    print('-'*20)
    

