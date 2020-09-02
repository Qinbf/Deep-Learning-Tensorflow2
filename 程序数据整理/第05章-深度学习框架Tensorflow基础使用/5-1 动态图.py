
# coding: utf-8

# In[4]:


import tensorflow as tf
# 创建一个常量
m1 = tf.constant([[4,4]])
# 创建一个常量
m2 = tf.constant([[2],[3]])
# 创建一个矩阵乘法，把m1和m2传入
product = tf.matmul(m1,m2)
# 打印结果
print(product)

