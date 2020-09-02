
# coding: utf-8

# In[14]:


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# 载入手写数字数据
digits = load_digits()
# 打印数据集的shape，行表示数据集个数，列表示每个数据的特征数
print('data shape:',digits.data.shape)
# 打印数据标签的shape，数据标签的值为0-9
print('target shape:',digits.target.shape)
# 准备显示第0张图片，图片为灰度图
plt.imshow(digits.images[0],cmap='gray')
# 显示图片
plt.show()

