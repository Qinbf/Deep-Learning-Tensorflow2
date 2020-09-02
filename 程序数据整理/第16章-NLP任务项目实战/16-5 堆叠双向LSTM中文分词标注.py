
# coding: utf-8

# In[20]:


import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
# pip install plot_model
from plot_model import plot_model
import json


# In[21]:


# 批次大小
batch_size = 256
# 训练周期
epochs = 30
# 词向量长度
embedding_dims = 128
# cell数量
lstm_cell = 64
# 最长的句子设置为128，只保留长度小于128的句子，最好不要截断句子
# 大部分的句子都是小于128长度的
max_length=128


# In[22]:


# 读入数据
# {b:begin, m:middle, e:end, s:single}，分别代表每个状态代表的是该字在词语中的位置，
# b代表该字是词语中的起始字，m代表是词语中的中间字，e代表是词语中的结束字，s则代表是单字成词
text = open('msr_train.txt', encoding='gb18030').read()
# 根据换行符切分数据
text = text.split('\n')


# In[23]:


# 得到所有的数据和标签
def get_data(s):
    # 匹配(.)/(.)格式的数据
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        # 返回数据和标签，0为数据，1为标签
        return s[:,0],s[:,1]

# 数据
data = []
# 标签
label = []
# 循环每个句子
for s in text:
    # 分离文字和标签
    d = get_data(s)
    if d:
        # 0为数据
        data.append(d[0])
        # 1为标签
        label.append(d[1])

# 存入DataFrame
df = pd.DataFrame(index=range(len(data)))
df['data'] = data
df['label'] = label
# 只保留长度小于max_length的句子
df = df[df['data'].apply(len) <= max_length]


# In[24]:


# 把data中所有的list都变成字符串格式
texts = [' '.join(x) for x in df['data']]


# In[25]:


# 实例化Tokenizer，设置字典中最大词汇数为num_words
# Tokenizer会自动过滤掉一些符号比如：!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n
tokenizer = Tokenizer()
# 传入我们的训练数据，建立词典，词的编号根据词频设定，频率越大，编号越小，
tokenizer.fit_on_texts(texts) 
# 把词转换为编号，编号大于num_words的词会被过滤掉
sequences = tokenizer.texts_to_sequences(texts) 
# 把序列设定为max_length的长度，超过max_length的部分舍弃，不到max_length则补0
# padding='pre'在句子前面进行填充，padding='post'在句子后面进行填充
X = pad_sequences(sequences, maxlen=max_length, padding='post')  


# In[26]:


# 把token_config保存到json文件中，模型预测阶段可以使用
file = open('token_config.json','w',encoding='utf-8')
# 把tokenizer变成json数据
token_config = tokenizer.to_json()
# 保存json数据
json.dump(token_config, file)


# In[27]:


# 计算字典中词的数量，由于有填充的词，所有加1
# 中文的单字词数量一般比较少，这个数据集只有5000多个词
num_words = len(tokenizer.index_word)+1


# In[28]:


# 相当于是把字符类型的标签变成了数字类型的标签
tag = {'o':0, 's':1, 'b':2, 'm':3, 'e':4}
Y = [] 
# 循环原来的标签
for label in df['label']:
    temp = []
    # 把sbme转变成1234
    temp = temp + [tag[l] for l in label]
    temp = temp + [0]*(max_length-len(temp))
    Y.append(temp)
Y = np.array(Y)


# In[29]:


# 切分数据集
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2)


# In[30]:


# 定义模型
sequence_input = Input(shape=(max_length))
# Embedding层，
# mask_zero=True，计算时忽略0值，也就是填充的数据不参与计算
embedding_layer = Embedding(num_words, embedding_dims, mask_zero=True)(sequence_input)
# 双向LSTM，因为我们的任务是分词标签，因此需要LSTM每个序列的Hidden State输出值
# return_sequences=True表示返回所有序列LSTM的输出，默认只返回最后一个序列LSTM的输出
x = Bidirectional(LSTM(lstm_cell, return_sequences=True))(embedding_layer)
# 堆叠多个双向LSTM
x = Bidirectional(LSTM(lstm_cell, return_sequences=True))(x)
x = Bidirectional(LSTM(lstm_cell, return_sequences=True))(x)
# TimeDistributed该包装器可以把一个层应用到输入的每一个时间步上
# 也就是说LSTM每个序列输出的Hidden State都应该连接一个Dense层并预测出5个结果
# 这5个结果分别对应：sbmeo。o为填充值，对应标签0。
preds = TimeDistributed(Dense(5, activation='softmax'))(x)
# 定义模型输入输出
model = Model(inputs=sequence_input, outputs=preds)
# 画图
plot_model(model)


# In[12]:


# 定义代价函数，优化器
# 使用sparse_categorical_crossentropy，标签不需要转变为独热编码one-hot
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
# 保存模型
model.save('lstm_tag.h5')

