
# coding: utf-8

# In[1]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import jieba
import numpy as np


# In[2]:


# 载入tokenizer
json_file = open('token_config.json','r',encoding='utf-8')
token_config = json.load(json_file)
tokenizer = tokenizer_from_json(token_config)


# In[3]:


# 载入模型
model = load_model('cnn_model.h5')


# In[4]:


# 情感预测
def predict(text):
    # 对句子分词
    cw = list(jieba.cut(text)) 
    # list转字符串，元素之间用' '隔开
    texts = ' '.join(cw)
    # 把词转换为编号，编号大于30000的词会被过滤掉
    sequences = tokenizer.texts_to_sequences([texts]) 
    # model.input_shape为(None, 202)，202为训练模型时的序列长度
    # 把序列设定为202的长度，超过202的部分舍弃，不到202则补0
    sequences = pad_sequences(sequences, maxlen=model.input_shape[1], padding='pre')
    # 模型预测
    result = np.argmax(model.predict(sequences))
    if(result==1):
        print("正面情绪")
    else:
        print("负面情绪")


# In[5]:


predict("今天阳光明媚，手痒想打球了。")


# In[6]:


predict("一大屋子人，结果清早告停水了，我崩溃到现在[抓狂]")

