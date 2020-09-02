
# coding: utf-8

# In[1]:


from tf2_bert.models import build_transformer_model
from tf2_bert.tokenizers import Tokenizer
import numpy as np


# In[2]:


# 定义预训练模型路径
model_dir = './chinese_roberta_wwm_ext_L-12_H-768_A-12'
# BERT参数
config_path = model_dir+'/bert_config.json'
# 保存模型权值参数的文件
checkpoint_path = model_dir+'/bert_model.ckpt'
# 词表
dict_path = model_dir+'/vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path) 
# 建立模型，加载权重
# with_mlm=True表示使用mlm的功能，模型结构及最后的输出会发生一些变化，可以用来预测被mask的token
model = build_transformer_model(config_path, checkpoint_path, with_mlm=True) 


# In[3]:


# 分词并转化为编码
token_ids, segment_ids = tokenizer.encode('机器学习是一门交叉学科')
# 把“学”字和“习”字变成“[MASK]”符号
token_ids[3] = token_ids[4] = tokenizer._token_dict['[MASK]']
# 增加一个维度表示批次大小为1
token_ids = np.expand_dims(token_ids,axis=0)
# 增加一个维度表示批次大小为1
segment_ids = np.expand_dims(segment_ids,axis=0)
# 传入模型进行预测
pre = model.predict([token_ids, segment_ids])[0]
# 我们可以看到第3，4个位置经过模型预测，[MASK]变成了“学习”
print(tokenizer.decode(pre[3:5].argmax(axis=1)))  


# In[4]:


# 分词并转化为编码
token_ids, segment_ids = tokenizer.encode('机器学习是一门交叉学科')
# 把“交”字和“叉”字变成“[MASK]”符号
token_ids[8] = token_ids[9] = tokenizer._token_dict['[MASK]']
# 增加一个维度表示批次大小为1
token_ids = np.expand_dims(token_ids,axis=0)
# 增加一个维度表示批次大小为1
segment_ids = np.expand_dims(segment_ids,axis=0)
# 传入模型进行预测
pre = model.predict([token_ids, segment_ids])[0]
# 我们可以看到第8，9个位置经过模型预测，[MASK]变成了“什么”，句子变成了一个疑问句
# 虽然模型没有预测出原始句子的词汇，不过作为完形填空，填入一个“什么”句子也是正确
print(tokenizer.decode(pre[8:10].argmax(axis=1)))  

