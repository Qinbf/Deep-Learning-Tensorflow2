
# coding: utf-8

# In[1]:


# 导入numpy 科学计算包
import numpy as np


# In[2]:


# 定义输入
x0 = 1
x1 = 0
x2 = -1
# 定义权值
w0 = -5
w1 = 0
w2 = 0
# 定义正确的标签
t = 1
# 定义学习率lr(learning rate)
lr = 1
# 定义偏置值
b = 0


# In[3]:


# 循环一个比较大的次数，比如100
for i in range(100):
    # 打印权值
    print(w0,w1,w2)
    # 计算感知器的输出
    y = np.sign(w0 * x0 + w1 * x1 + w2*x2)
    # 如果感知器输出不等于正确的标签
    if(y != t):
        # 更新权值
        w0 = w0 + lr * (t-y) * x0
        w1 = w1 + lr * (t-y) * x1
        w2 = w2 + lr * (t-y) * x2
    # 如果感知器输出等于正确的标签
    else:
        # 训练结束
        print('done')
        # 退出循环
        break

