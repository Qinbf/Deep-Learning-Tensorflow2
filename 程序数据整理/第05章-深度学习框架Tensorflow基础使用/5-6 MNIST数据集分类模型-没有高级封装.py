
# coding: utf-8

# In[9]:


import tensorflow as tf


# In[10]:


# 载入数据集
mnist = tf.keras.datasets.mnist
# 载入数据，数据载入的时候就已经划分好训练集和测试集
# 训练集数据x_train的数据形状为（60000，28，28）
# 训练集标签y_train的数据形状为（60000）
# 测试集数据x_test的数据形状为（10000，28，28）
# 测试集标签y_test的数据形状为（10000）
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 对训练集和测试集的数据进行归一化处理，有助于提升模型训练速度
x_train, x_test = x_train / 255.0, x_test / 255.0
# 把训练集和测试集的标签转为独热编码
y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)


# In[11]:


# 创建dataset对象，使用dataset对象来管理数据
mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# 训练周期设置为1（把所有训练集数据训练一次称为训练一个周期）
mnist_train = mnist_train.repeat(1)
# 批次大小设置为32（每次训练模型传入32个数据进行训练）
mnist_train = mnist_train.batch(32)

# 创建dataset对象，使用dataset对象来管理数据
mnist_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# 训练周期设置为1（把所有训练集数据训练一次称为训练一个周期）
mnist_test = mnist_test.repeat(1)
# 批次大小设置为32（每次训练模型传入32个数据进行训练）
mnist_test = mnist_test.batch(32)


# In[12]:


# 模型定义
# 先用Flatten把数据从3维变成2维，(60000,28,28)->(60000,784)
# 设置输入数据形状input_shape不需要包含数据的数量，（28,28）即可
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(10, activation='softmax')
])
# 优化器定义
optimizer = tf.keras.optimizers.SGD(0.1)
# 计算平均值
train_loss = tf.keras.metrics.Mean(name='train_loss')
# 训练准确率计算
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
# 计算平均值
test_loss = tf.keras.metrics.Mean(name='test_loss')
# 测试准确率计算
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


# In[13]:


# 我们可以用@tf.function装饰器来将python代码转成tensorflow的图表示代码，用于加速代码运行速度
# 定义一个训练模型的函数
@tf.function
def train_step(data, label):
    # 固定写法，使用tf.GradientTape()来计算梯度
    with tf.GradientTape() as tape:
        # 传入数据获得模型预测结果
        predictions = model(data)
        # 对比label和predictions计算loss
        loss = tf.keras.losses.MSE(label, predictions)
        # 传入loss和模型参数，计算权值调整
        gradients = tape.gradient(loss, model.trainable_variables)
        # 进行权值调整
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # 计算平均loss
        train_loss(loss)
        # 计算平均准确率
        train_accuracy(label, predictions)
    
# 我们可以用@tf.function装饰器来将python代码转成tensorflow的图表示代码，用于加速代码运行速度
# 定义一个模型测试的函数
@tf.function
def test_step(data, label):
    # 传入数据获得模型预测结果
    predictions = model(data)
    # 对比label和predictions计算loss
    t_loss = tf.keras.losses.MSE(label, predictions)
    # 计算平均loss
    test_loss(t_loss)
    # 计算平均准确率
    test_accuracy(label, predictions)


# In[14]:


# 训练10个周期（把所有训练集数据训练一次称为训练一个周期）
EPOCHS = 10

for epoch in range(EPOCHS):
    # 训练集循环60000/32=1875次
    for image, label in mnist_train:
        # 每次循环传入一个批次的数据和标签训练模型
        train_step(image, label)
    # 测试集循环10000/32=312.5->313次
    for test_image, test_label in mnist_test:
        # 每次循环传入一个批次的数据和标签进行测试
        test_step(test_image, test_label)
  
    # 打印结果
    template = 'Epoch {}, Loss: {:.3}, Accuracy: {:.3}, Test Loss: {:.3}, Test Accuracy: {:.3}'
    print(template.format(epoch+1,
                          train_loss.result(), 
                          train_accuracy.result(),
                          test_loss.result(), 
                          test_accuracy.result()))
  

