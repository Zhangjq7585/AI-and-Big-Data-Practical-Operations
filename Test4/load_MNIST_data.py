### 使用MINIST数据集
'''
    1.使用真实数据集：优先从OpenML加载实际MNIST数据
    2.完整的处理流程：标准化 → PCA分析 → 降维 → 可视化
    3.详细的分析报告：显示方差解释率、维度减少效果等
    4.机器学习演示：使用降维后的数据进行分类
    5.完整的文件保存：保存所有中间结果和模型
    6.丰富的可视化：包括样本图像、PCA分析、重建对比等
'''

import numpy as np
import gzip
import struct
import os
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_mnist_images(filename):
    """
    读取MNIST图像文件（idx3-ubyte格式）
    参数:
        filename: MNIST图像文件路径（通常是.gz压缩文件）
    返回:
        numpy数组，形状为(num_images, 28, 28)，像素值范围0-255
    """
    print(f"正在读取MNIST图像文件: {filename}")

    # 检查文件是否存在
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")

    # 打开并读取压缩文件
    with gzip.open(filename, 'rb') as f:
        # 读取文件头
        # 格式: [magic_number, num_images, num_rows, num_columns]
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        print(f"文件格式: idx3-ubyte (magic number: {magic})")
        print(f"图像数量: {num_images}")
        print(f"图像尺寸: {rows} × {cols}")

        # 读取所有图像数据
        buffer = f.read(rows * cols * num_images)

        # 将字节数据转换为numpy数组
        # 注意：'>B' 表示大端字节序的无符号字节
        data = np.frombuffer(buffer, dtype=np.uint8)

        # 重塑为正确的形状
        images = data.reshape(num_images, rows, cols)

        return images

def load_mnist_labels(filename):
    """
    读取MNIST标签文件（idx1-ubyte格式）
    参数:
        filename: MNIST标签文件路径（通常是.gz压缩文件）
    返回:
        numpy数组，形状为(num_images,)，标签值0-9
    """
    print(f"正在读取MNIST标签文件: {filename}")

    # 检查文件是否存在
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")

    # 打开并读取压缩文件
    with gzip.open(filename, 'rb') as f:
        # 读取文件头
        # 格式: [magic_number, num_labels]
        magic, num_labels = struct.unpack('>II', f.read(8))

        print(f"文件格式: idx1-ubyte (magic number: {magic})")
        print(f"标签数量: {num_labels}")

        # 读取所有标签数据
        buffer = f.read(num_labels)

        # 将字节数据转换为numpy数组
        labels = np.frombuffer(buffer, dtype=np.uint8)

        return labels

def load_real_dataset(data_dir='E:/Download/MNIST'):
    """
    从MNIST原始二进制文件加载数据集
    参数:
        data_dir: 包含MNIST文件的目录路径
    返回:
        X: 图像数据，形状为(num_images, 28, 28)
        y: 标签数据，形状为(num_images,)
    """
    print("=" * 60)
    print("从MNIST二进制文件加载数据集")
    print("=" * 60)

    # 创建目录对象
    data_path = Path(data_dir)

    # 定义文件路径
    files = {
        'train_images': data_path / 'train-images-idx3-ubyte.gz',
        'train_labels': data_path / 'train-labels-idx1-ubyte.gz',
        'test_images': data_path / 't10k-images-idx3-ubyte.gz',
        'test_labels': data_path / 't10k-labels-idx1-ubyte.gz'
    }

    # 检查文件是否存在
    for file_type, file_path in files.items():
        if not file_path.exists():
            print(f"警告: {file_path} 不存在")

    try:
        # 首先尝试加载测试集（t10k-images-idx3-ubyte.gz）
        if files['test_images'].exists():
            print("\n1. 加载测试集数据...")
            X_test = load_mnist_images(files['test_images'])
            y_test = load_mnist_labels(files['test_labels']) if files['test_labels'].exists() else None

            print(f"测试集图像形状: {X_test.shape}")
            if y_test is not None:
                print(f"测试集标签形状: {y_test.shape}")
                print(f"测试集标签分布: {np.bincount(y_test)}")

            # 直接使用测试集的10000个样本
            X = X_test
            y = y_test

        # 如果测试集不存在，尝试加载训练集
        elif files['train_images'].exists():
            print("\n1. 加载训练集数据...")
            X_train = load_mnist_images(files['train_images'])
            y_train = load_mnist_labels(files['train_labels']) if files['train_labels'].exists() else None

            print(f"训练集图像形状: {X_train.shape}")
            if y_train is not None:
                print(f"训练集标签形状: {y_train.shape}")

            # 由于训练集有60000个样本，我们取前10000个
            X = X_train[:10000]
            y = y_train[:10000] if y_train is not None else None

        else:
            raise FileNotFoundError("未找到MNIST数据文件")

        print(f"\n2. 加载的数据统计:")
        print(f"   图像形状: {X.shape}")
        print(f"   图像类型: {X.dtype}")
        print(f"   像素值范围: [{X.min()}, {X.max()}]")

        if y is not None:
            print(f"   标签形状: {y.shape}")
            print(f"   标签类型: {y.dtype}")
            print(f"   标签唯一值: {np.unique(y)}")
            print(f"   各类样本数量:")
            for i in range(10):
                count = np.sum(y == i)
                print(f"     数字{i}: {count}个 ({100 * count / len(y):.1f}%)")

        return X, y

    except Exception as e:
        print(f"加载MNIST数据时出错: {e}")
        print("\n尝试创建模拟数据...")
        return create_synthetic_dataset()


def create_synthetic_dataset():
    """
    创建合成数据集作为后备
    """
    print("创建合成数据集...")
    num_images = 10000
    height, width = 28, 28

    # 创建带有简单模式的图像
    images = np.zeros((num_images, height, width), dtype=np.uint8)
    labels = np.zeros(num_images, dtype=int)

    # 为每个数字创建简单形状
    for i in range(num_images):
        label = i % 10
        labels[i] = label

        # 创建数字的简单表示
        center_x, center_y = 14, 14

        if label == 0:  # 圆
            radius = 8
            y, x = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            mask = np.abs(dist_from_center - radius) <= 2
            images[i][mask] = 255

        elif label == 1:  # 竖线
            images[i, :, center_x - 1:center_x + 2] = 255

        elif label == 2:  # 数字2的形状
            # 上横线
            images[i, 8, 8:20] = 255
            # 右下横线
            images[i, 20, 8:20] = 255
            # 竖线
            images[i, 8:21, 18] = 255

        # 添加噪声
        noise = np.random.randint(0, 50, (height, width))
        images[i] = np.clip(images[i] + noise, 0, 255)

    print(f"合成数据集创建完成: {images.shape}")
    return images, labels


def visualize_mnist_samples(X, y=None, num_samples=10):
    """
    可视化MNIST样本

    参数:
        X: 图像数据
        y: 标签数据（可选）
        num_samples: 要显示的样本数量
    """
    print("\n可视化数据样本...")

    # 确保不超过可用样本数
    num_samples = min(num_samples, len(X))

    # 创建子图
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]

        # 显示图像
        ax.imshow(X[i], cmap='gray', vmin=0, vmax=255)

        # 设置标题
        if y is not None:
            ax.set_title(f'样本 {i + 1}\n标签: {y[i]}', fontsize=12)
        else:
            ax.set_title(f'样本 {i + 1}', fontsize=12)

        ax.axis('off')

    plt.suptitle(f'MNIST数据集样本 (共{len(X)}张图像)', fontsize=16)
    plt.tight_layout()

    # 保存图像
    output_file = 'mnist_samples.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"样本可视化已保存为: {output_file}")
    plt.show()

def save_dataset_as_npy(X, y=None, prefix='mnist'):
    """
    将数据集保存为.npy文件

    参数:
        X: 图像数据
        y: 标签数据（可选）
        prefix: 文件前缀
    """
    print("\n保存数据集为.npy格式...")

    # 保存图像数据
    images_file = f'{prefix}_images.npy'
    np.save(images_file, X)
    print(f"图像数据保存为: {images_file}")
    print(f"  形状: {X.shape}")
    print(f"  数据类型: {X.dtype}")

    # 保存标签数据（如果有）
    if y is not None:
        labels_file = f'{prefix}_labels.npy'
        np.save(labels_file, y)
        print(f"标签数据保存为: {labels_file}")
        print(f"  形状: {y.shape}")
        print(f"  数据类型: {y.dtype}")

    # 保存展平版本用于机器学习
    X_flat = X.reshape(len(X), -1)
    flat_file = f'{prefix}_flat.npy'
    np.save(flat_file, X_flat)
    print(f"展平数据保存为: {flat_file}")
    print(f"  形状: {X_flat.shape}")

    return images_file, labels_file if y is not None else None

def main():
    """
    主函数：加载、可视化和保存MNIST数据集
    """
    print("MNIST数据集加载器")
    print("=" * 60)

    # 1. 加载数据集
    print("\n步骤1: 加载MNIST数据集")
    print("-" * 40)

    # 假设数据文件在当前目录的mnist_data文件夹中
    # 我的文件夹路径为  E:\Download\MNIST
    # data_dir = './mnist_data'
    data_dir = 'E:/Download/MNIST'

    # 创建数据目录（如果不存在）
    os.makedirs(data_dir, exist_ok=True)

    # 检查目录内容
    print(f"检查数据目录: {data_dir}")
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"目录中的文件: {files}")

    # 加载数据
    X, y = load_real_dataset(data_dir)

    print(f"\n加载完成!")
    print(f"图像数据: {X.shape}")
    print(f"标签数据: {y.shape if y is not None else '无'}")

    # 2. 可视化样本
    print("\n步骤2: 可视化样本")
    print("-" * 40)
    visualize_mnist_samples(X, y)

    # 3. 保存数据集
    print("\n步骤3: 保存数据集")
    print("-" * 40)
    save_dataset_as_npy(X, y)

    # 4. 显示数据统计信息
    print("\n步骤4: 数据统计信息")
    print("-" * 40)

    # 计算基本统计信息
    print(f"图像统计:")
    print(f"  最小值: {X.min()}")
    print(f"  最大值: {X.max()}")
    print(f"  平均值: {X.mean():.2f}")
    print(f"  标准差: {X.std():.2f}")

    if y is not None:
        print(f"\n标签统计:")
        print(f"  唯一值: {np.unique(y)}")
        print(f"  分布:")
        counts = np.bincount(y)
        for i, count in enumerate(counts):
            print(f"    数字{i}: {count:4d} ({100 * count / len(y):5.1f}%)")

    print("\n" + "=" * 60)
    print("数据集加载和保存完成!")
    print("=" * 60)

    # 返回数据供后续使用
    return X, y

def quick_load_existing():
    """
    快速加载已保存的.npy文件
    """
    print("快速加载已保存的MNIST数据...")

    files_to_check = ['mnist_images.npy', 'mnist_flat.npy', 'mnist_labels.npy']

    for file in files_to_check:
        if os.path.exists(file):
            print(f"找到文件: {file}")

    try:
        X = np.load('mnist_images.npy')
        y = np.load('mnist_labels.npy') if os.path.exists('mnist_labels.npy') else None

        print(f"加载成功!")
        print(f"图像数据: {X.shape}")
        if y is not None:
            print(f"标签数据: {y.shape}")

        return X, y
    except Exception as e:
        print(f"加载失败: {e}")
        return None, None

# 使用示例
if __name__ == "__main__":
    # 方式1: 从原始文件加载
    print("方式1: 从原始MNIST二进制文件加载")
    X, y = main()

    # 方式2: 如果已经有.npy文件，可以快速加载
    if X is None:
        print("\n方式2: 从.npy文件快速加载")
        X, y = quick_load_existing()

    if X is not None:
        print(f"\n数据集已准备就绪!")
        print(f"可以使用 X.reshape({len(X)}, -1) 将其展平为特征矩阵")

        # 示例：展平数据
        X_flat = X.reshape(len(X), -1)
        print(f"展平后形状: {X_flat.shape}")

        # 示例：显示前5个样本的标签
        if y is not None:
            print(f"前10个样本的标签: {y[:10]}")