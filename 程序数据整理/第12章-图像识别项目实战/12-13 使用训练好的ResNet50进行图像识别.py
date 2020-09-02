
# coding: utf-8

# In[1]:


from tensorflow.keras.applications.resnet50 import ResNet50
# imagenet数据处理工具
from tensorflow.keras.applications.imagenet_utils import decode_predictions,preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import matplotlib.pyplot as plt
import os
import numpy as np


# In[2]:


# 图片大小
image_size = 224
# 存放测试图片的文件夹
image_dir = 'test'
# 载入使用imagenet训练好的预训练模型
# include_top=True表示模型包含全连接层
# include_top=False表示模型不包含全连接层
# 下载的模型会存放在你的用户目录下.keras隐藏文件夹下的models文件夹中
resnet50 = ResNet50(weights='imagenet',include_top=True, input_shape=(image_size,image_size,3))


# In[4]:


# 循环目录下的图片并进行显示预测
for file in os.listdir(image_dir):
    # 测试图片完整路径
    file_dir = os.path.join(image_dir,file)
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
    # 把像素数值归一化为(-1,1)之间，并让RGB通道减去对应均值
    x = preprocess_input(x) 
    # preds.shap->(1, 1000),1000个概率值
    preds = resnet50.predict(x)
    # decode_predictions用于预测结果解码
    # 将测试结果解码为如下形式：
    # [(编码1, 英文名称1, 概率1),(编码2, 英文名称2, 概率2)...]
    # top=1表示概率最大的1个结果，top=3表示概率最大的3个结果
    predicted_classes = decode_predictions(preds, top=1)
    imagenet_id, name, confidence = predicted_classes[0][0]
    # 打印结果
    print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))

