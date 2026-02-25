##使用 TensorFlow/Keras 实现了 CIFAR-10 数据集的本地加载、数据增强、CNN 模型构建、训练、评估以及测试集预测结果输出


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


# 设置matplotlib正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题

# -------------------- 1. 加载本地 CIFAR-10 数据集 --------------------
def load_cifar10_batch(file):
    """加载单个 CIFAR-10 batch 文件"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data = dict[b'data']
    labels = dict[b'labels']
    # 图像数据形状为 (10000, 3072)，转换为 (10000, 32, 32, 3)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, np.array(labels)

def load_local_cifar10(data_dir):
    """从指定目录加载 CIFAR-10 数据（包含 data_batch_1..5 和 test_batch）"""
    # 训练集：合并 5 个 batch
    x_train, y_train = [], []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(batch_file)
        x_train.append(data)
        y_train.append(labels)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    # 测试集
    test_file = os.path.join(data_dir, 'test_batch')
    x_test, y_test = load_cifar10_batch(test_file)

    return (x_train, y_train), (x_test, y_test)

# 请将此处路径修改为您存放 CIFAR-10 数据集的文件夹（例如 './cifar-10-batches-py'）
data_dir = 'E:/Download/cifar-10-batches-py'
(x_train, y_train), (x_test, y_test) = load_local_cifar10(data_dir)

# 类别名称（CIFAR-10 官方分类）
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f'训练集大小: {x_train.shape}, 测试集大小: {x_test.shape}')

# -------------------- 2. 数据预处理 --------------------
# 归一化像素值到 [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 将标签转换为 one-hot 编码
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# -------------------- 3. 数据增强 --------------------
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
# 对于验证集/测试集通常只做归一化，不增强
test_datagen = ImageDataGenerator()

# 注意：fit 时将使用 flow 方法，我们会在训练时传递原始数据
train_generator = train_datagen.flow(x_train, y_train_cat, batch_size=64)
validation_generator = test_datagen.flow(x_test, y_test_cat, batch_size=64)

# -------------------- 4. 构建 CNN 模型 --------------------
model = models.Sequential([
    # 第一个卷积块
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # 第二个卷积块
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # 第三个卷积块
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    # 全连接部分
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()

# -------------------- 5. 编译模型 --------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------- 6. 训练模型 --------------------
history = model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // 64,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=len(x_test) // 64,
    verbose=1
)

# -------------------- 7. 绘制训练曲线 --------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves.png')  # 保存图片
plt.show()

# -------------------- 8. 评估模型 --------------------
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f'测试集准确率: {test_acc:.4f}')

# -------------------- 9. 对测试集进行预测并输出结果 --------------------
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# 生成结果文件（CSV 格式）
# 第一列是图像索引（从0开始），第二列是预测的类别名称
output_file = 'cifar10_predictions.csv'
with open(output_file, 'w') as f:
    f.write('id,label\n')
    for idx, pred in enumerate(predicted_classes):
        f.write(f'{idx},{class_names[pred]}\n')

print(f'预测结果已保存至 {output_file}')

# （可选）展示部分预测结果
def plot_sample_predictions(images, true_labels, pred_labels, class_names, num_samples=10):
    plt.figure(figsize=(15, 6))
    indices = np.random.choice(len(images), num_samples, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[idx])
        true_name = class_names[true_labels[idx]]
        pred_name = class_names[pred_labels[idx]]
        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        plt.title(f'True: {true_name}\nPred: {pred_name}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_sample_predictions(x_test, y_test, predicted_classes, class_names)

