
# coding: utf-8

# In[1]:


# 安装验证码生成库:pip install captcha
from captcha.image import ImageCaptcha  
import random
import string

# 字符包含所有数字和所有大小写英文字母，一共62个
characters = string.digits+string.ascii_letters

# 随机产生验证码，长度为4
def random_captcha_text(char_set=characters, captcha_size=4):
    # 验证码列表
    captcha_text = []
    for i in range(captcha_size):
        # 随机选择
        c = random.choice(char_set)
        # 加入验证码列表
        captcha_text.append(c)
    return captcha_text
 
# 生成字符对应的验证码
def gen_captcha_text_and_image():
    # 验证码图片宽高可以设置，默认width=160, height=60
    image = ImageCaptcha(width=160, height=60)
    # 获得随机生成的验证码
    captcha_text = random_captcha_text()
    # 把验证码列表转为字符串
    captcha_text = ''.join(captcha_text)
    # 保存验证码图片
    image.write(captcha_text, 'captcha/' + captcha_text + '.jpg')


# 产生1000次随机验证码
# 真正的数量可能会少于1000
# 因为重名的图片会被覆盖掉
num = 1000
for i in range(num):
    gen_captcha_text_and_image()

print("生成完毕")

