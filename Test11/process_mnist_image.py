## 以下是将本地 MNIST ubyte.gz 文件转换为图片并保存到文件夹

import os
import gzip
import numpy as np
from PIL import Image   # pip install numpy Pillow
# ------------------------------
# 配置路径
# ------------------------------
MNIST_GZ_DIR = r"E:\Download\MNIST"  # 原始 gz 文件所在目录
OUTPUT_DIR = r"E:\Download\mnist_PNG"  # 图片输出根目录

# MNIST 文件名称
FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz'
}

# ------------------------------
# 读取 gz 文件并返回图像和标签
# ------------------------------
def load_mnist(data_path, files_dict):
    """加载 MNIST 的四个 gz 文件，返回 (x_train, y_train), (x_test, y_test)"""
    # 训练标签
    with gzip.open(os.path.join(data_path, files_dict['train_labels']), 'rb') as f:
        y_train = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    # 测试标签
    with gzip.open(os.path.join(data_path, files_dict['test_labels']), 'rb') as f:
        y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    # 训练图像
    with gzip.open(os.path.join(data_path, files_dict['train_images']), 'rb') as f:
        x_train = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    # 测试图像
    with gzip.open(os.path.join(data_path, files_dict['test_images']), 'rb') as f:
        x_test = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    return (x_train, y_train), (x_test, y_test)


# ------------------------------
# 保存图片到对应类别的子文件夹
# ------------------------------
def save_images(images, labels, split_name):
    """
    将图像数组保存为图片文件
    split_name: 'train' 或 'test'
    """
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # 为每个类别创建子文件夹
    for digit in range(10):
        class_dir = os.path.join(split_dir, str(digit))
        os.makedirs(class_dir, exist_ok=True)

    total = len(images)
    for idx, (img_array, label) in enumerate(zip(images, labels)):
        # 创建 PIL Image 对象
        img = Image.fromarray(img_array, mode='L')  # 'L' 为灰度模式
        # 构造保存路径：/split_name/label/index.png
        filename = f"{idx:06d}.png"  # 6位数字，不足补零
        filepath = os.path.join(split_dir, str(label), filename)
        img.save(filepath)

        if (idx + 1) % 5000 == 0:
            print(f"  [{split_name}] 已保存 {idx + 1}/{total} 张图片")


# ------------------------------
# 主程序
# ------------------------------
def main():
    print("正在从本地 gz 文件加载 MNIST 数据集...")
    (x_train, y_train), (x_test, y_test) = load_mnist(MNIST_GZ_DIR, FILES)
    print(f"训练集: {x_train.shape[0]} 张图片")
    print(f"测试集: {x_test.shape[0]} 张图片")

    print("开始保存训练集图片...")
    save_images(x_train, y_train, "train")

    print("开始保存测试集图片...")
    save_images(x_test, y_test, "test")

    print("所有图片保存完成！")
    print(f"图片根目录: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()

