
# coding: utf-8

# In[1]:


# 这个程序我是在Tensorflow1的环境中运行的
import tensorflow as tf
# 创建一个常量
m1 = tf.constant([[4,4]])
# 创建一个常量
m2 = tf.constant([[2],[3]])
# 创建一个矩阵乘法，把m1和m2传入
product = tf.matmul(m1,m2)
# Tensorflow1的程序跟一般的python程序不太一样
# 这个时候打印product，只能看到product的属性，不能计算它的值
# 应该这里我只定义了计算图，图必须在会话中运行，我们还没有定义会话
print(product)


# In[2]:


# 定义一个会话
sess = tf.Session()
# 调用sess的run方法来执行矩阵乘法
# 计算product，最终计算的结果存放在result中
result = sess.run(product)
print(result)
# 关闭会话
sess.close()

