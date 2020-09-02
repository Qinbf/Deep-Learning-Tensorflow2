
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
model = build_transformer_model(config_path, checkpoint_path) 


# In[3]:


# 句子0
sentence0 = '机器学习'
# 句子1
sentence1 = '深度学习'
# 用分词器对句子分词
tokens = tokenizer.tokenize(sentence0)
# 分词后自动在句子前加上[CLS]，在句子后加上[SEP]
print(tokens)


# In[4]:


# 编码测试
token_ids, segment_ids = tokenizer.encode(sentence0)
# [CLS]的编号为101，机为3322，器为1690，学为2110，习为739，[SEP]为102
print('token_ids:',token_ids)
# 因为只有一个句子所以segment_ids都是0
print('segment_ids:',segment_ids)


# In[5]:


# 编码测试
token_ids, segment_ids = tokenizer.encode(sentence0,sentence1)
# 可以看到两个句子分词后的结果为：
# ['[CLS]', '机', '器', '学', '习', '[SEP]', '深', '度', '学', '习', [SEP]]
print('token_ids:',token_ids)
# 0表示第一个句子的token，1表示第二个句子的token
print('segment_ids:',segment_ids)


# In[6]:


# 增加一个维度表示批次大小为1
token_ids = np.expand_dims(token_ids,axis=0)
# 增加一个维度表示批次大小为1
segment_ids = np.expand_dims(segment_ids,axis=0)


# In[7]:


# 传入模型进行预测
pre = model.predict([token_ids, segment_ids])
# 得到的结果中1表示批次大小，11表示11个token，768表示特征向量长度
# 这里就是把句子的token转化为了特征向量
print(pre.shape)

