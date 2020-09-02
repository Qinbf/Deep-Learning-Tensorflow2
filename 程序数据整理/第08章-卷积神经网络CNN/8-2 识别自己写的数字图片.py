
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# In[4]:


# 载入数据
mnist = tf.keras.datasets.mnist
# 载入数据，数据载入的时候就已经划分好训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 获取一张照片，并把它的shape变成二维（784->28×28）,用灰度图显示
plt.imshow(x_train[18],cmap='gray')
# 不显示坐标
plt.axis('off')
plt.show()


# In[6]:


# 载入我自己写的数字图片
img=Image.open('6.jpg')
# 显示图片
plt.imshow(img)
# 不显示坐标
plt.axis('off')
plt.show()


# In[7]:


# 把图片大小变成28×28，并且把它从3D的彩色图变为1D的灰度图
image = np.array(img.resize((28,28)).convert('L'))
# 显示图片,用灰度图显示
plt.imshow(image,cmap='gray')
# 不显示坐标
plt.axis('off')
plt.show()


# In[8]:


# 观察发现我自己写的数字是白底黑字，MNIST数据集的图片是黑底白字
# 所以我们需要先把图片从白底黑字变成黑底白字，就是255-image
# MNIST数据集的数值都是0-1之间的，所以我们还需要/255.0对数值进行归一化
image = (255-image)/255.0
# 显示图片，用灰度图显示
plt.imshow(image,cmap='gray')
# 不显示坐标
plt.axis('off')
plt.show()


# In[13]:


# 把数据处理变成4维数据
image = image.reshape((1,28,28,1))
# 载入训练好的模型
model = load_model('mnist.h5')
# predict_classes对数据进行预测并得到它的类别
prediction = model.predict_classes(image)
print(prediction)

