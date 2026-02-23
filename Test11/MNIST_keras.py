## 使用 Keras 高级 API（tf.keras.Sequential）实现

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

    with gzip.open(os.path.join(data_path, files['train_labels']), 'rb') as f:
        y_train = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with gzip.open(os.path.join(data_path, files['test_labels']), 'rb') as f:
        y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

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

# ------------------------------
# 3. 使用 Sequential 构建 CNN 模型
# 该实现使用 model.compile() + model.fit() 进行训练
# 模型结构：输入层(28*28*1)->卷积层1(32个3*3卷积核，ReLU激活)->
# 池化层1(2*2最大池化)->卷积层2(64个3*3卷积核，ReLU激活)->
# 池化层2(2*2最大池化)->扁平化层->全连接层(128个神经元，ReLU激活)
# 输出层(10个神经元，Softmax激活)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# ------------------------------
# 4. 编译模型
# ------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ------------------------------
# 5. 训练模型
# ------------------------------
epochs = 5
batch_size = 32
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,   # 从训练集拆分10%作为验证集
                    verbose=1)

# ------------------------------
# 6. 在测试集上评估
# ------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n测试集准确率: {test_acc:.4f}")

