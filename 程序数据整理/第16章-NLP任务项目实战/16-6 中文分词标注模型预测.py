
# coding: utf-8

# In[1]:


import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


# In[2]:


# 句子长度，需要跟模型训练时一致
max_length = 128


# In[3]:


# 载入tokenizer
json_file = open('token_config.json','r',encoding='utf-8')
token_config = json.load(json_file)
tokenizer = tokenizer_from_json(token_config)
# 获得字典，键为字，值为编号
word_index = tokenizer.word_index


# In[4]:


# 载入模型
model = load_model('lstm_tag.h5')


# In[5]:


# 载入数据集做处理主要是为了计算状态转移概率
# 读入数据
text = open('msr_train.txt', encoding='gb18030').read()
# 根据换行符切分数据
text = text.split('\n')

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


# In[6]:


# texts二维数据，一行一个句子
# 比如ngrams(texts,2,2)，只计算2-grams
# 比如ngrams(texts,2,4)，计算2-grams，3-grams，4-grams
def ngrams(texts, MIN_N, MAX_N):
    # 定义空字典记录
    ngrams_dict = {}
    # 循环每一个句子
    for tokens in texts:
        # 计算一个句子token数量
        n_tokens = len(tokens)
        # 词汇组合统计
        for i in range(n_tokens):
            for j in range(i+MIN_N, min(n_tokens, i+MAX_N)+1):
                # 词汇组合list转字符串
                temp = ''.join(tokens[i:j])
                # 字典计数加一
                ngrams_dict[temp] = ngrams_dict.get(temp, 0) + 1
    # 返回字典
    return ngrams_dict


# In[7]:


# 统计状态转移次数
ngrams_dict = ngrams(label,2,2)
print(ngrams_dict)


# In[8]:


# 计算状态转移总次数
sum_num = 0
for value in ngrams_dict.values():
    sum_num = sum_num + value


# In[9]:


# 计算状态转移概率
p_sb = ngrams_dict['sb']/sum_num
p_be = ngrams_dict['be']/sum_num
p_es = ngrams_dict['es']/sum_num
p_ss = ngrams_dict['ss']/sum_num
p_bm = ngrams_dict['bm']/sum_num
p_me = ngrams_dict['me']/sum_num
p_mm = ngrams_dict['mm']/sum_num
p_eb = ngrams_dict['eb']/sum_num
# p_oo用于表示不可能的转移，-np.inf负无穷
p_oo = -np.inf

# 使用条件随机场CRF来计算转移矩阵有可能效果会更好
# 这里我们用简单的二元模型来定义状态转移矩阵
# oo,os,ob,om,oe,
# so,ss,sb,sm,se
# bo,bs,bb,bm,be
# mo,ms,mb,mm,me
# eo,es,eb,em,ee
# 其中sm,se,bs,bb,ms,mb,em,ee这几个状态转移是不存在的
# o为填充状态，跟o相关的转移也都不需要考虑
transition_params = [[p_oo,p_oo,p_oo,p_oo,p_oo],
                     [p_oo,p_ss,p_sb,p_oo,p_oo],
                     [p_oo,p_oo,p_oo,p_bm,p_be],
                     [p_oo,p_oo,p_oo,p_mm,p_me],
                     [p_oo,p_es,p_eb,p_oo,p_oo]]


# In[10]:


# 维特比算法
def viterbi_decode(sequence, transition_params):
    """
    Args:
      sequence: 一个[seq_len, num_tags]矩阵
      transition_params: 一个[num_tags, num_tags]矩阵
    Returns:
      viterbi: 一个[seq_len]序列
    """
    # 假设状态转移共有num_tags种状态
    # 创建一个跟sequence相同形状的网格
    score = np.zeros_like(sequence)
    # 创建一个跟sequence相同形状的path，用于记录路径
    path = np.zeros_like(sequence, dtype=np.int32)
    # 起始分数
    score[0] = sequence[0]
    for t in range(1, sequence.shape[0]):
        # t-1时刻score得分加上trans分数，得到下一时刻所有状态转移[num_tags, num_tags]的得分
        T = np.expand_dims(score[t - 1], 1) + transition_params
        # t时刻score = 计算每个状态转移的最大得分 + 下个序列预测得分
        score[t] = np.max(T, 0) + sequence[t] 
        # 记录每个状态转移的最大得分所在位置 
        path[t] = np.argmax(T, 0)
    # score[-1]为最后得到的num_tags种状态得分
    # np.argmax(score[-1])找到最高分数所在位置
    viterbi = [np.argmax(score[-1])]
    # 回头确定来的路径，相当于知道最高分以后从后往前走
    for p in reversed(path[1:]):
        viterbi.append(p[viterbi[-1]])
    # 反转viterbi列表，把viterbi变成正向路径
    viterbi.reverse()
    # 计算最大得分，如果需要可以return
    # viterbi_score = np.max(score[-1])
    return viterbi


# In[39]:


# 小句分词函数
def cut(sentence):
    # 如果句子大于最大长度，只取max_length个词
    if len(sentence) >= max_length:
        seq = sentence[:max_length]
    # 如果不足max_length，则填充
    else:
        seq = []
        for s in sentence:
            try:
                # 在字典里查询编号
                seq.append(word_index[s])
            except:
                # 如果不在字典里填充0
                seq.append(0)
        seq = seq + [0]*(max_length-len(sentence))
    # 获得预测结果，shape(32,5)
    preds = model.predict([seq])[0]
    # 维特比算法
    viterbi = viterbi_decode(preds, transition_params)
    # 只保留跟句子相同长度的分词标注
    y = viterbi[:len(sentence)]
    # 分词
    words = []
    for i in range(len(sentence)):
        # 如果标签为s或b，append到结果的list中
        if y[i] in [1, 2]:
            words.append(sentence[i])
        else:
        # 如果标签为m或e，在list最后一个元素中追加内容
            words[-1] += sentence[i]
    return  words


# In[40]:


# 根据符号断句
cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!()（）]')
# 先分小句，再对小句分词
def cut_word(s):
    result = []
    # 指针设置为0
    i = 0
    # 根据符号断句
    for c in cuts.finditer(s):
        # 对符号前的部分分词
        result.extend(cut(s[i:c.start()]))
        # 加入符号
        result.append(s[c.start():c.end()])
        # 移动指针到符号后面
        i = c.end()
    # 对最后的部分进行分词
    result.extend(cut(s[i:]))
    return result


# In[41]:


print(cut_word('针对新冠病毒感染，要做好“早发现、早报告、早隔离、早治疗”，及时给予临床治疗的措施。'))


# In[45]:


print(cut_word('广义相对论是描写物质间引力相互作用的理论'))


# In[15]:


print(cut_word('阿尔法围棋（AlphaGo）是第一个击败人类职业围棋选手、第一个战胜围棋世界冠军的人工智能，是谷歌（Google）旗下DeepMind公司戴密斯·哈萨比斯领衔的团队开发。'))

