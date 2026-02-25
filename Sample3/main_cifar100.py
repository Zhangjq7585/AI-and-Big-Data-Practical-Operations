#针对 CIFAR-100 数据集,使用 TensorFlow/Keras 实现完整的图像分类流程，

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 设置matplotlib正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题

# -------------------- 1. 加载本地 CIFAR-100 数据集 --------------------
def load_cifar100_batch(file):
    """加载单个 CIFAR-100 batch 文件"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 假设数据集解压在 'cifar-100-python' 文件夹中，路径请根据实际情况修改
data_dir = 'E:/Download/cifar-100-python'  # 请修改为你的实际路径
train_file = os.path.join(data_dir, 'train')
test_file = os.path.join(data_dir, 'test')

train_dict = load_cifar100_batch(train_file)
test_dict = load_cifar100_batch(test_file)

X_train = train_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
y_train = np.array(train_dict[b'fine_labels'])  # 使用细粒度标签（100类）

X_test = test_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
y_test = np.array(test_dict[b'fine_labels'])

print(f'训练集形状: {X_train.shape}, 标签数: {len(np.unique(y_train))}')
print(f'测试集形状: {X_test.shape}')

# -------------------- 2. 数据预处理与增强 --------------------
# 归一化
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 将标签转换为 one-hot 编码
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

# 数据增强
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# -------------------- 3. 构建 CNN 模型 --------------------
def create_cnn_model(input_shape=(32, 32, 3), num_classes=100):
    model = models.Sequential([
        # 数据增强层（仅在训练时生效）
        layers.Input(shape=input_shape),
        data_augmentation,

        # 第一个卷积块
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # 第二个卷积块
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # 第三个卷积块
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # 全局平均池化代替全连接层，减少过拟合
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = create_cnn_model()
model.summary()

# -------------------- 4. 编译与训练 --------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 回调函数：早停和学习率衰减
callbacks_list = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    callbacks.ModelCheckpoint('best_cifar100_model.h5', save_best_only=True)
]

# 训练模型
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=100,  # 由于早停，实际不会跑满
    validation_data=(X_test, y_test),
    callbacks=callbacks_list,
    verbose=1
)

# -------------------- 5. 绘制准确率和损失曲线 --------------------
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='train_accuracy')
    ax1.plot(history.history['val_accuracy'], label='val_accuracy')
    ax1.set_title('Accuracy over epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='train_loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_title('Loss over epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

plot_training_history(history)

# -------------------- 6. 对测试集进行预测并输出结果文件 --------------------
# 加载最佳模型
best_model = tf.keras.models.load_model('best_cifar100_model.h5')

# 预测测试集（这里使用 X_test，如果比赛要求对另外的测试集，请替换）
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # 转换为类别索引

# 保存为 CSV 文件（假设 id 为测试集样本索引）
import pandas as pd
output_df = pd.DataFrame({
    'id': np.arange(len(y_pred_classes)),
    'label': y_pred_classes
})
output_df.to_csv('test_predictions.csv', index=False)
print("预测结果已保存至 test_predictions.csv")

