
# coding: utf-8

# In[1]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import numpy as np


# 常用的一些数据增强策略   
# * rotation_range是一个0~180的度数，用来指定随机选择图片的角度。  
# * width_shift和height_shift用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比  
# * rescale值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。  
# * shear_range是用来进行错切变换的程度
# * zoom_range用来进行随机的放大  
# * horizontal_flip随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候  
# * brightness_range亮度变化，取值范围[0,1,10]  
# * fill_mode用来指定当需要进行像素填充，如旋转，水平和竖直位移时，如何填充新出现的像素  

# In[2]:


datagen = ImageDataGenerator(
    rotation_range = 40,     # 随机旋转度数
    width_shift_range = 0.2, # 随机水平平移
    height_shift_range = 0.2,# 随机竖直平移
    rescale = 1/255,         # 数据归一化
    shear_range = 30,       # 随机错切变换
    zoom_range = 0.2,        # 随机放大
    horizontal_flip = True,  # 水平翻转
    brightness_range = (0.7,1.3), # 亮度变化
    fill_mode = 'nearest',   # 填充方式
) 


# In[3]:


# 载入图片
img = load_img('image.jpg')
# 把图片变成array，此时数据是3维
# 3维(height,width,channel)
x = img_to_array(img)
# 在第0个位置增加一个维度
# 我们需要把数据变成4维，然后再做数据增强
# 4维(1,height,width,channel)
x = np.expand_dims(x,0)


# In[4]:


# 生成20张图片
i = 0
# 生成的图片都保存在temp文件夹中，文件名前缀为new_cat,图片格式为jpeg
for batch in datagen.flow(x, batch_size=1, save_to_dir='temp', save_prefix='new_cat', save_format='jpeg'):
    i += 1
    if i==20:
        break

