
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


# 载入数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 归一化
x_train, x_test = x_train / 255.0, x_test / 255.0
# 标签转独热编码
y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)


# In[3]:


# 创建dataset对象
mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# 训练周期
mnist_train = mnist_train.repeat(1)
# 批次大小
mnist_train = mnist_train.batch(32)
# 创建dataset对象
mnist_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# 训练周期
mnist_test = mnist_test.repeat(1)
# 批次大小
mnist_test = mnist_test.batch(32)


# In[4]:


# 模型定义
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(10, activation='softmax')
])
# 优化器定义
optimizer = tf.keras.optimizers.SGD(0.1)
# 训练loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
# 训练准确率计算
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
# 测试loss
test_loss = tf.keras.metrics.Mean(name='test_loss')
# 测试准确率计算
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


# In[5]:


# 模型训练
@tf.function
def train_step(data, label):
    with tf.GradientTape() as tape:
        # 传入数据预测结果
        predictions = model(data)
        # 计算loss
        loss = tf.keras.losses.MSE(label, predictions)
        # 计算权值调整
        gradients = tape.gradient(loss, model.trainable_variables)
        # 进行权值调整
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # 计算平均loss
        train_loss(loss)
        # 计算平均准确率
        train_accuracy(label, predictions)
    
# 模型测试
@tf.function
def test_step(data, label):
    # 传入数据预测结果
    predictions = model(data)
    # 计算loss
    t_loss = tf.keras.losses.MSE(label, predictions)
    # 计算平均loss
    test_loss(t_loss)
    # 计算平均准确率
    test_accuracy(label, predictions)


# In[6]:


# 定义Checkpoint，用于保存优化器和模型参数
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
# restore载入Checkpoint
# latest_checkpoint表示载入编号最大的Checkpoint
ckpt.restore(tf.train.latest_checkpoint('tf2_ckpts/'))
# 载入模型后继续训练
EPOCHS = 5
# 训练5个周期
for epoch in range(EPOCHS):
    # 循环60000/32=1875次
    for image, label in mnist_train:
        # 训练模型
        train_step(image, label)
    # 循环10000/32=312.5->313次
    for test_image, test_label in mnist_test:
        # 测试模型
        test_step(test_image, test_label)
  
    # 打印结果
    template = 'Epoch {}, Loss: {:.3}, Accuracy: {:.3}, Test Loss: {:.3}, Test Accuracy: {:.3}'
    print (template.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result(),
                         test_loss.result(), 
                         test_accuracy.result()))

