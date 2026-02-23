## 使用 TensorFlow 底层 API
## 两层卷积+池化，两层全连接，输出层使用 softmax
import tensorflow as tf
import numpy as np
import gzip
import os

# ------------------------------
# 1. 从本地 gz 文件加载 MNIST 数据集
# ------------------------------
def load_mnist(data_path):
    """从指定目录加载 MNIST 的四个 gz 文件，返回训练和测试数据"""
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    # 读取标签
    with gzip.open(os.path.join(data_path, files['train_labels']), 'rb') as f:
        y_train = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with gzip.open(os.path.join(data_path, files['test_labels']), 'rb') as f:
        y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    # 读取图像
    with gzip.open(os.path.join(data_path, files['train_images']), 'rb') as f:
        x_train = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(os.path.join(data_path, files['test_images']), 'rb') as f:
        x_test = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)

    return (x_train, y_train), (x_test, y_test)

# ------------------------------
# 2. 数据预处理
# ------------------------------
DATA_PATH = r"E:\Download\MNIST"
(x_train, y_train), (x_test, y_test) = load_mnist(DATA_PATH)

# 归一化并增加通道维度
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)    # (10000, 28, 28, 1)

# 划分验证集（从训练集取10%）
val_size = 6000
x_val, y_val = x_train[:val_size], y_train[:val_size]
x_train, y_train = x_train[val_size:], y_train[val_size:]

# 转换为 tf.data.Dataset，提高训练效率
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# ------------------------------
# 3. 构建 CNN 模型（继承 tf.keras.Model）
# ------------------------------
class CNNModel(tf.keras.Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D(2)
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

model = CNNModel()

# ------------------------------
# 4. 定义损失函数、优化器和评估指标
# ------------------------------
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# ------------------------------
# 5. 自定义训练循环
# ------------------------------
epochs = 5
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # 训练阶段
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc_metric.update_state(y_batch, logits)

    # 每个 epoch 结束后显示训练准确率
    train_acc = train_acc_metric.result()
    print(f"训练准确率: {train_acc:.4f}")
    train_acc_metric.reset_states()

    # 验证阶段
    for x_batch, y_batch in val_dataset:
        logits = model(x_batch, training=False)
        val_acc_metric.update_state(y_batch, logits)
    val_acc = val_acc_metric.result()
    print(f"验证准确率: {val_acc:.4f}")
    val_acc_metric.reset_states()

# ------------------------------
# 6. 在测试集上评估
# ------------------------------
for x_batch, y_batch in test_dataset:
    logits = model(x_batch, training=False)
    test_acc_metric.update_state(y_batch, logits)
test_acc = test_acc_metric.result()
print(f"\n测试集准确率: {test_acc:.4f}")