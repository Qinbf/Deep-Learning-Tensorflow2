
# coding: utf-8

# In[1]:


from tf2_bert.models import build_transformer_model
from tf2_bert.tokenizers import Tokenizer
from tensorflow.keras.models import load_model
import numpy as np


# In[2]:


# 载入模型
model = load_model('bert_model.h5')
# 词表路径
dict_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12'+'/vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path) 


# In[3]:


# 预测函数
def predict(text):
    # 分词并把token变成编号，句子长度需要与模型训练时一致
    token_ids, segment_ids = tokenizer.encode(text, first_length=256)
    # 增加一个维度表示批次大小为1
    token_ids = np.expand_dims(token_ids,axis=0)
    # 增加一个维度表示批次大小为1
    segment_ids = np.expand_dims(segment_ids,axis=0)
    # 模型预测
    pre = model.predict([token_ids, segment_ids])
    # 去掉一个没用的维度
    pre = np.array(pre).reshape((7,3))
    # 获得可能性最大的预测结果
    pre = np.argmax(pre,axis=1)
    
    comment = ''
    if(pre[0]==0):
        comment += '性价比差,'
    elif(pre[0]==1):
        comment += '-,'
    elif(pre[0]==2):
        comment += '性价比好,'

    if(pre[1]==0):
        comment += '质量差,'
    elif(pre[1]==1):
        comment += '-,'
    elif(pre[1]==2):
        comment += '质量好,'

    if(pre[2]==0):
        comment += '希望有活动,'
    elif(pre[2]==1):
        comment += '-,'
    elif(pre[2]==2):
        comment += '参加了活动,'

    if(pre[3]==0):
        comment += '客服物流包装差,'
    elif(pre[3]==1):
        comment += '-,'
    elif(pre[3]==2):
        comment += '客服物流包装好,'

    if(pre[4]==0):
        comment += '新用户,'
    elif(pre[4]==1):
        comment += '-,'
    elif(pre[4]==2):
        comment += '老用户,'

    if(pre[5]==0):
        comment += '不会再买,'
    elif(pre[5]==1):
        comment += '-,'
    elif(pre[5]==2):
        comment += '会继续购买,'

    if(pre[6]==0):
        comment += '差评'
    elif(pre[6]==1):
        comment += '中评'
    elif(pre[6]==2):
        comment += '好评'
        
    return pre,comment


# In[4]:


pre,comment = predict("还没用，不知道怎么样")
print('pre:',pre)
print('comment:',comment)


# In[5]:


pre,comment = predict("质量不错，还会再来，价格优惠")
print('pre:',pre)
print('comment:',comment)


# In[6]:


pre,comment = predict("好用不贵物美价廉，用后皮肤水水的非常不错")
print('pre:',pre)
print('comment:',comment)


# In[7]:


pre,comment = predict('一直都用这款产品，便宜又补水，特别好用，今后要一直屯下去。')
print('pre:',pre)
print('comment:',comment)


# In[8]:


pre,comment = predict('趁着搞活动又囤了几盒，很划算，天天用也不心疼，补水效果还可以的')
print('pre:',pre)
print('comment:',comment)


# In[9]:


pre,comment = predict('我周六买的，星期一才发货，问客服没有回复，不过速度还是快，星期二收到的。发货速度有待改进。')
print('pre:',pre)
print('comment:',comment)


# In[10]:


pre,comment = predict('人生中第一次差评，差评一是给这个产品，用了过敏；二是给这个客服，说过敏仅支持退货并且运费自理。我的天！那我就不退了吧。只能说自己倒霉咯，过敏了没人管，退货还得自掏腰包，最惨不过我')
print('pre:',pre)
print('comment:',comment)


# In[11]:


pre,comment = predict('自从朋友推荐就一直使用这款面膜，哈哈哈哈，这款面膜一件用了很久了，每次活动买，比较实惠划算，比较适合我自己。唯一感觉不足的就是乳液太少。发货也特别快，值得购买。会在买的。')
print('pre:',pre)
print('comment:',comment)

