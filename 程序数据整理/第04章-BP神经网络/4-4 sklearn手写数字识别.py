
# coding: utf-8

# In[1]:


# 载入BP神经网络算法
from sklearn.neural_network import MLPClassifier 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# In[2]:


#载入数据
digits = load_digits()
#数据
x_data = digits.data 
#标签
y_data = digits.target 
# X中原来的数值范围是0-255之间，归一化后变成0-1之间
x_data -= x_data.min()
x_data /= x_data.max() - x_data.min()
# 分割数据1/4为测试数据，3/4为训练数据
# 有1347个训练数据，450个测试数据
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.25) 


# In[3]:


# 定义神经网络模型，模型输入神经元个数和输出神经元个数不需要设置
# hidden_layer_sizes用于设置隐藏层结构：
# 比如(50)表示有1个隐藏层，隐藏层神经元个数为50
# 比如(100,20)表示有2个隐藏层，第1个隐藏层有100个神经元，第2个隐藏层有20个神经元
# 比如(100,20,10)表示3个隐藏层，神经元个数分别为100，20，10
# max_iter设置训练次数
mlp = MLPClassifier(hidden_layer_sizes=(100,20), max_iter=500)
# fit传入训练集数据开始训练模型
mlp.fit(x_train,y_train)


# In[4]:


# predict用于模型预测
predictions = mlp.predict(x_test)
# 标签数据和模型预测数据进行对比，计算分类评估指标
print(classification_report(y_test, predictions))

