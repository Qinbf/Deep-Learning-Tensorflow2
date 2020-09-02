
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt

# 输入数据
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
# 标签
T = np.array([[0],
              [1],
              [1],
              [0]])

# 定义一个2层的神经网络：2-10-1
# 输入层2个神经元，隐藏层10个神经元，输出层1个神经元
# 输入层到隐藏层的权值初始化，2行10列
W1 = np.random.random([2,10])
# 隐藏层到输出层的权值初始化，10行1列
W2 = np.random.random([10,1])
# 初始化偏置值，偏置值的初始化一般可以取0，或者一个比较小的常数，如0.1
# 隐藏层的10个神经元偏置
b1 = np.zeros([10])
# 输出层的1个神经元偏置
b2 = np.zeros([1])
# 学习率设置
lr = 0.1
# 定义训练周期数
epochs = 100001
# 定义测试周期数
test = 5000

# 定义sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 定义sigmoid函数导数
def dsigmoid(x):
    return x*(1-x)

# 更新权值和偏置值
def update():
    global X,T,W1,W2,lr,b1,b2
    
    # 隐藏层输出
    L1 = sigmoid(np.dot(X,W1) + b1)
    # 输出层输出
    L2 = sigmoid(np.dot(L1,W2) + b2)
    
    # 求输出层的学习信号
    delta_L2 = (T - L2) * dsigmoid(L2)
    # 隐藏层的学习信号
    delta_L1 = delta_L2.dot(W2.T) * dsigmoid(L1)
    
    # 求隐藏层到输出层的权值改变
    # 由于一次计算了多个样本，所以需要求平均
    delta_W2 = lr * L1.T.dot(delta_L2) / X.shape[0]
    # 输入层到隐藏层的权值改变
    # 由于一次计算了多个样本，所以需要求平均
    delta_W1 = lr * X.T.dot(delta_L1) / X.shape[0]
    
    # 更新权值
    W2 = W2 + delta_W2
    W1 = W1 + delta_W1
    
    # 改变偏置值
    # 由于一次计算了多个样本，所以需要求平均
    b2 = b2 + lr * np.mean(delta_L2, axis=0)
    b1 = b1 + lr * np.mean(delta_L1, axis=0)

# 定义空list用于保存loss
loss = []
# 训练模型
for i in range(epochs):
    # 更新权值
    update()
    # 每训练5000次计算一次loss值
    if i % test == 0:
        # 隐藏层输出
        L1 = sigmoid(np.dot(X,W1) + b1)
        # 输出层输出
        L2 = sigmoid(np.dot(L1,W2) + b2)
        # 计算loss值
        print('epochs:',i,'loss:',np.mean(np.square(T - L2) / 2))
        # 保存loss值
        loss.append(np.mean(np.square(T - L2) / 2))

# 画图训练周期数与loss的关系图
plt.plot(range(0,epochs,test),loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
        
# 隐藏层输出
L1 = sigmoid(np.dot(X,W1) + b1)
# 输出层输出
L2 = sigmoid(np.dot(L1,W2) + b2)
print('output:')
print(L2)

# 因为最终的分类只有0和1，所以我们可以把
# 大于等于0.5的值归为1类，小于0.5的值归为0类
def predict(x):
    if x>=0.5:
        return 1
    else:
        return 0

# map会根据提供的函数对指定序列做映射
# 相当于依次把L2中的值放到predict函数中计算
# 然后打印出结果
print('predict:')
for i in map(predict,L2):
    print(i)


