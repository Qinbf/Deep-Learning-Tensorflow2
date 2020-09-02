
# coding: utf-8

# In[1]:


import tensorflow as tf

# 定义一个变量
x = tf.Variable([1,2])
# 定义一个常量
a = tf.constant([3,3])
# 减法op
sub = tf.subtract(x, a)
# 加法op
add = tf.add(x,sub)

print(sub)
print(add)

