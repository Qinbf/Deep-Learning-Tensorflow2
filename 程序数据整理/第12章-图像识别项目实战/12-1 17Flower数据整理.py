
# coding: utf-8

# In[3]:


import os
import shutil

# 新建文件夹用于存放整理后的图片
os.mkdir('new_17_flowers')
for i in range(17):
    # 17个种类新建17个文件夹0-16
    os.mkdir('new_17_flowers'+'/'+'flower'+str(i))
    
# 循环所有花的图片    
for i,path in enumerate(os.listdir('17flowers/jpg/')):
    # 定义花的图片完整路径
    image_path = '17flowers/jpg/' + path
    # 复制到对应类别，每个类别80张图片
    shutil.copyfile(image_path, 'new_17_flowers'+'/'+'flower'+str(i//80)+'/'+path)

