
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image


# In[2]:


# 设置最长的一条边的长度
max_dim = 800
# 内容图片路径
content_path = '臭臭.jpeg'
# 风格图片路径
style_path = 'starry_night.jpg'
# 风格权重
style_weight=10
# 内容权重
content_weight=1
# 全变差正则权重
total_variation_weight=1e5
# 训练次数
stpes = 301
# 是否保存训练过程中产生的图片
save_img = True


# In[3]:


# 载入图片
def load_img(path_to_img):
    # 读取文件内容
    img = tf.io.read_file(path_to_img)
    # 变成3通道图片数据
    img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
#     img = tf.image.convert_image_dtype(img, tf.float32)
    # 获得图片高度和宽度，并转成float类型
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    # 最长的边的长度
    long_dim = max(shape)
    # 图像缩放，把图片最长的边变成max_dim
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    # resize图片大小
    img = tf.image.resize(img, new_shape)
    # 增加1个维度，变成4维数据
    img = img[tf.newaxis, :]
    return img

# 用于显示图片
def imshow(image, title=None):
    # 如图是4维度数据
    if len(image.shape) > 3:
        # 去掉size为1的维度如(1,300,300,3)->(300,300,3)
        image = tf.squeeze(image)
    # 显示图片
    plt.imshow(image)
    if title:
        # 设置图片title
        plt.title(title)
    plt.axis('off')
    plt.show()


# In[4]:


# 载入内容图片
content_image = load_img(content_path)
# 载入风格图片
style_image = load_img(style_path)
# 显示内容图片
imshow(content_image, 'Content Image')
# 显示风格图片
imshow(style_image, 'Style Image')


# In[5]:


# 用于计算content loss
# 这里只取了一层的输出进行对比，取多层输出效果变化不大
content_layers = ['block5_conv2'] 

# 用于计算风格的卷积层
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

# 计算层数
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# In[6]:


# 创建一个新模型，输入与vgg16一样，输出为指定层的输出
def vgg_layers(layer_names):
    # 载入VGG16的卷积层部分
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    # VGG16的模型参数不参与训练
    vgg.trainable = False
    # 获取指定层的输出值
    outputs = [vgg.get_layer(name).output for name in layer_names]
    # 定义一个新的模型，输入与vgg16一样，输出为指定层的输出
    model = tf.keras.Model([vgg.input], outputs)
    # 返回模型
    return model


# In[7]:


# 获得输出风格层特征的模型
style_extractor = vgg_layers(style_layers)
# 图像预处理，主要是减去颜色均值，RGB转BGR
preprocessed_input = tf.keras.applications.vgg16.preprocess_input(style_image*255)
# 风格图片传入style_extractor，提取风格层的输出
style_outputs = style_extractor(preprocessed_input)


# In[8]:


# Gram矩阵的计算
def gram_matrix(input_tensor):
    # 爱因斯坦求和，bijc表示input_tensor中的4个维度，bijd表示input_tensor中的4个维度
    # 例如input_tensor的shape为(1,300,200,32)，那么b=1,i=300,j=200,c=32,d=32
    # ->bcd表示计算后得到的数据维度为(1,32,32),得到的结果表示特征图与特征图之间的相关性
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    # 特征图的shape
    input_shape = tf.shape(input_tensor)
    # 特征图的高度乘以宽度得到特征值数量
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    # 除以特征值的数量
    return result/(num_locations)


# In[9]:


# 构建一个返回风格特征和内容特征的模型
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        # 获得输出风格层和内容层特征的模型
        self.vgg =  vgg_layers(style_layers + content_layers)
        # 用于计算风格的卷积层
        self.style_layers = style_layers
        # 用于计算content loss的卷积层
        self.content_layers = content_layers
        # 风格层的数量
        self.num_style_layers = len(style_layers)

    def call(self, inputs):
        # 图像预处理，主要是减去颜色均值，RGB转BGR 
        preprocessed_input = tf.keras.applications.vgg16.preprocess_input(inputs*255.0)
        # 图片传入模型，提取风格层和内容层的输出
        outputs = self.vgg(preprocessed_input)
        # 获得风格特征输出和内容特征输出
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])
        # 计算风格特征的Gram矩阵
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        # 把风格特征的Gram矩阵分别存入字典
        style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
        # 把内容特征存入字典
        content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
        # 返回结果
        return {'content':content_dict, 'style':style_dict}


# In[10]:


# 构建一个返回风格特征和内容特征的模型
extractor = StyleContentModel(style_layers, content_layers)
# 计算得到风格图片的风格特征
style_targets = extractor(style_image)['style']
# 计算得到内容图片的内容特征
content_targets = extractor(content_image)['content']


# In[11]:


# 初始化要训练的图片
image = tf.Variable(content_image)
# 定义优化器
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
# 把数值范围限制在0-1之间
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# In[12]:


# 定义风格和内容loss
def style_content_loss(outputs):
    # 模型输出的风格特征
    style_outputs = outputs['style']
    # 模型输出的内容特征
    content_outputs = outputs['content']
    # 计算风格loss
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    # 计算内容loss
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    # 风格加内容loss
    loss = style_loss + content_loss
    return loss

# 施加全变差正则，全变差正则化常用于图片去噪，可以使生成的图片更加平滑自然
def total_variation_loss(image):
    x_deltas = image[:,:,1:,:] - image[:,:,:-1,:]
    y_deltas = image[:,1:,:,:] - image[:,:-1,:,:]
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)


# In[13]:


# 我们可以用@tf.function装饰器来将python代码转成tensorflow的图表示代码，用于加速代码运行速度
@tf.function()
# 定义一个训练模型的函数
def train_step(image):
    # 固定写法，使用tf.GradientTape()来计算梯度
    with tf.GradientTape() as tape:
        # 传入图片获得风格特征和内容特征
        outputs = extractor(image)
        # 计算风格和内容loss
        loss = style_content_loss(outputs)
        # 再加上全变差正则loss
        loss += total_variation_weight*total_variation_loss(image)
    # 传入loss和模型参数，计算权值调整
    grad = tape.gradient(loss, image)
    # 进行权值调整，这里要调整的权值就是image图像的像素值
    opt.apply_gradients([(grad, image)])
    # 把数值范围限制在0-1之间
    image.assign(clip_0_1(image))


# In[14]:


# 训练steps次
for n in range(stpes):
    # 训练模型
    train_step(image)
    # 每训练5次打印一次图片
    if n%5==0:
        imshow(image.read_value(), "Train step: {}".format(n))
        # 保存图片
        if save_img==True:
            # 去掉一个维度
            s_image = tf.squeeze(image)
            # 把array变成Image对象
            s_image = Image.fromarray(np.uint8(s_image.numpy()*255))
            # 设置保存路径保存图片
            s_image.save('temp/'+'steps_'+str(n)+'.jpg')

