##########  说明  ########
'''
一、数据转换步骤
    1. MNIST数据加载
        MNIST数据集的idx3-ubyte格式是二进制文件，包含像素数据
        每张图像为28×28像素的灰度图，共784个像素点
        10000张图像需要转换为10000×784的特征矩阵
    2. 数据预处理步骤
        加载原始二进制数据：读取idx3-ubyte文件
        数据解析：解析文件头，提取图像数据
        形状转换：从10000×28×28转换为10000×784
        标准化：将像素值从[0,255]缩放到[0,1]或标准化
        降维处理：使用PCA保留95%的方差信息

二、降维方法说明
    PCA（主成分分析）
    原理：通过线性变换将原始特征投影到新的坐标系中
    目标：找到数据方差最大的方向作为主成分
    优势：
    保留主要信息的同时减少特征维度
    消除特征间的相关性
    计算效率高，适合图像数据
    信息保留：通过设置n_components=0.95，保留95%的方差信息
'''

import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import struct

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

'''
label_file = 't10k-labels-idx1-ubyte.gz'  # 标签文件
        with gzip.open(label_file, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
'''

def load_idx3_ubyte(filename):
    """
    加载MNIST的idx3-ubyte格式文件
    参数:
        filename: 文件路径
    返回:
        numpy数组: 图像数据
    """
    print(f"正在加载文件: {filename}")
    # 打开并读取文件
    with gzip.open(filename, 'rb') as f:
        # 读取文件头信息
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        print(f"文件头信息: 魔数={magic}, 图像数量={num_images}, 行数={rows}, 列数={cols}")

        # 读取所有图像数据
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)

        # 重塑为图像形状
        images = data.reshape(num_images, rows, cols)

    print(f"成功加载 {num_images} 张图像，每张 {rows}x{cols} 像素")
    return images

def preprocess_data(images):
    """
    数据预处理
    参数:
        images: 原始图像数据
    返回:
        X_scaled: 标准化后的特征矩阵
        scaler: 标准化器对象
    """
    print("\n=== 数据预处理 ===")
    # 1. 重塑为特征矩阵 (10000×784)
    print("1. 将图像数据重塑为特征矩阵...")
    X = images.reshape(images.shape[0], -1)
    print(f"   原始形状: {images.shape} -> 特征矩阵形状: {X.shape}")
    print(f"   每个样本的特征数: {X.shape[1]}")

    # 2. 标准化数据
    print("\n2. 数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float32))
    print("   标准化完成!")

    # 显示统计信息
    print(f"\n   标准化后数据统计:")
    print(f"   均值: {X_scaled.mean():.4f}, 标准差: {X_scaled.std():.4f}")
    print(f"   最小值: {X_scaled.min():.4f}, 最大值: {X_scaled.max():.4f}")

    return X_scaled, scaler

def apply_pca(X_scaled, variance_ratio=0.95):
    """
    应用PCA降维
    参数:
        X_scaled: 标准化后的特征矩阵
        variance_ratio: 保留的方差比例
    返回:
        X_pca: 降维后的特征矩阵
        pca: PCA对象
    """
    print("\n=== PCA降维处理 ===")

    # 1. 创建并拟合PCA模型
    print(f"1. 创建PCA模型，保留 {variance_ratio *100}% 的方差信息...")
    pca = PCA(n_components=variance_ratio, random_state=42)

    # 2. 应用PCA
    print("2. 应用PCA降维...")
    X_pca = pca.fit_transform(X_scaled)

    # 3. 显示降维结果
    print("\n3. PCA降维结果:")
    print(f"   原始维度: {X_scaled.shape[1]}")
    print(f"   降维后维度: {X_pca.shape[1]}")
    print(f"   压缩比: {(X_pca.shape[1 ] /X_scaled.shape[1 ] *100):.2f}%")

    # 4. 显示各主成分解释的方差比例
    print(f"\n4. 主成分方差解释比例:")
    print(f"   累计解释方差比例: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"   主成分数量: {pca.n_components_}")

    # 前几个主成分的解释方差比例
    n_components_to_show = min(10, pca.n_components_)
    print(f"\n   前{n_components_to_show}个主成分的解释方差比例:")
    for i in range(n_components_to_show):
        print(f"   主成分 { i +1}: {pca.explained_variance_ratio_[i]:.4f}")

    return X_pca, pca

def visualize_pca_results(pca,X_scaled, scaler, n_samples=5):
    """
    可视化PCA结果
    参数:
        pca: 训练好的PCA对象
    """
    print("\n=== PCA结果可视化 ===")

    # 1.计算累计解释方差
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    print("1. 累计解释方差:")
    for i in range(min(20, len(cumulative_variance))):
        print(f"   主成分 { i +1:2d}: {cumulative_variance[i]:.4f}")

    # 找到达到95%方差所需的主成分数
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\n2. 达到95%方差所需的主成分数: {n_components_95}")

    # 2. 可视化原始图像和重建图像
    print("\n3. 可视化原始图像和重建图像...")

    # 随机选择n_samples个样本
    np.random.seed(42)
    sample_indices = np.random.choice(X_scaled.shape[0], n_samples, replace=False)
    # 创建图形
    plt.figure(figsize=(15, 20))

    # 使用网格布局
    gs = gridspec.GridSpec(5, n_samples)

    # 子图1：原始图像
    ax1 = plt.subplot(gs[0, :])
    ax1.set_title('原始图像 (标准化前)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 子图2：重建图像
    ax2 = plt.subplot(gs[1, :])
    ax2.set_title(f'PCA重建图像 (使用前{n_components_95}个主成分)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 子图3：标准化后的图像
    ax3 = plt.subplot(gs[2, :])
    ax3.set_title('标准化后图像', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # 子图4：PCA变换后的数据
    ax4 = plt.subplot(gs[3, :])
    ax4.set_title(f'PCA特征 (前{min(10, n_components_95)}个主成分)', fontsize=14, fontweight='bold')

    # 子图5：方差解释率曲线
    ax5 = plt.subplot(gs[4, :])

    # 准备数据用于重建
    X_pca_all = pca.transform(X_scaled)

    # 选择几个样本进行重建
    for i, idx in enumerate(sample_indices):
        # 获取原始标准化数据
        sample_scaled = X_scaled[idx].reshape(1, -1)

        # 获取PCA变换后的数据
        sample_pca = X_pca_all[idx]

        # 重建图像（使用全部主成分进行逆变换）
        sample_reconstructed = pca.inverse_transform(sample_pca.reshape(1, -1))

        # 反标准化重建后的数据
        sample_reconstructed_orig = scaler.inverse_transform(sample_reconstructed)

        # 反标准化原始数据用于显示
        sample_orig = scaler.inverse_transform(sample_scaled)

        # 归一化到0-255范围用于显示
        def normalize_to_uint8(data):
            data_reshaped = data.reshape(28, 28)
            # 如果数据有负值，先归一化到0-1
            if data_reshaped.min() < 0:
                data_reshaped = (data_reshaped - data_reshaped.min()) / (data_reshaped.max() - data_reshaped.min())
            # 缩放到0-255并转换为uint8
            data_reshaped = np.clip(data_reshaped * 255, 0, 255).astype(np.uint8)
            return data_reshaped

        # 转换数据
        original_img = normalize_to_uint8(sample_orig)
        reconstructed_img = normalize_to_uint8(sample_reconstructed_orig)
        normalized_img = normalize_to_uint8(sample_scaled)

        # 显示原始图像
        ax_img1 = plt.subplot(gs[0, i])
        ax_img1.imshow(original_img, cmap='gray')
        ax_img1.set_title(f'样本 {i + 1}', fontsize=10)
        ax_img1.axis('off')

        # 显示重建图像
        ax_img2 = plt.subplot(gs[1, i])
        ax_img2.imshow(reconstructed_img, cmap='gray')
        ax_img2.set_title(f'重建 {i + 1}', fontsize=10)
        ax_img2.axis('off')

        # 显示标准化后的图像
        ax_img3 = plt.subplot(gs[2, i])
        ax_img3.imshow(normalized_img, cmap='gray')
        ax_img3.set_title(f'标准化 {i + 1}', fontsize=10)
        ax_img3.axis('off')

    # 显示PCA特征（前几个主成分的值）
    pca_features = X_pca_all[sample_indices]
    n_features_to_show = min(10, pca_features.shape[1])

    # 创建特征值条形图
    x_pos = np.arange(n_features_to_show)
    width = 0.15  # 条形宽度

    for i in range(n_samples):
        ax4.bar(x_pos + i * width, pca_features[i, :n_features_to_show],
                width=width, label=f'样本{i + 1}', alpha=0.7)

    ax4.set_xlabel('主成分', fontsize=12)
    ax4.set_ylabel('特征值', fontsize=12)
    ax4.set_title(f'前{n_features_to_show}个主成分的特征值', fontsize=12)
    ax4.set_xticks(x_pos + width * (n_samples - 1) / 2)
    ax4.set_xticklabels([f'PC{i + 1}' for i in range(n_features_to_show)])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 3. 绘制方差解释率曲线
    print("4. 绘制方差解释率曲线...")
    # 设置颜色
    colors = plt.cm.viridis(np.linspace(0, 1, 3))

    # 绘制累计解释方差曲线
    ax5.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             linewidth=3, color=colors[0], label='累计解释方差')

    # 绘制单个主成分的解释方差
    individual_variance = pca.explained_variance_ratio_
    ax5.bar(range(1, len(individual_variance) + 1), individual_variance,
            alpha=0.5, color=colors[1], label='单个主成分方差')

    # 标记95%方差阈值线
    ax5.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95%方差阈值')

    # 标记达到95%方差点
    ax5.axvline(x=n_components_95, color='green', linestyle=':',
                linewidth=2, label=f'主成分数={n_components_95}')

    # 标记点
    ax5.scatter([n_components_95], [0.95], color='red', s=100, zorder=5)
    ax5.text(n_components_95, 0.92, f'({n_components_95}, 0.95)',fontsize=10, ha='center')

    # 计算信息压缩率
    original_dim = X_scaled.shape[1]
    compressed_dim = n_components_95
    compression_rate = (1 - compressed_dim / original_dim) * 100

    # 添加信息压缩率注释
    ax5.text(0.02, 0.98,
             f'原始维度: {original_dim}\n'
             f'压缩后维度: {compressed_dim}\n'
             f'压缩率: {compression_rate:.1f}%',
             transform=ax5.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax5.set_xlabel('主成分数量', fontsize=12)
    ax5.set_ylabel('解释方差比例', fontsize=12)
    ax5.set_title('PCA方差解释率分析', fontsize=14, fontweight='bold')
    ax5.set_xlim(1, min(100, len(cumulative_variance)))
    ax5.set_ylim(0, 1.05)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='lower right')

    # 计算重建误差
    print("\n5. 计算重建误差...")

    # 选择更多样本计算平均重建误差
    n_error_samples = 100
    error_indices = np.random.choice(X_scaled.shape[0], n_error_samples, replace=False)
    total_mse = 0

    for idx in error_indices:
        sample_scaled = X_scaled[idx].reshape(1, -1)
        sample_pca = X_pca_all[idx]
        sample_reconstructed = pca.inverse_transform(sample_pca.reshape(1, -1))

        # 计算均方误差
        mse = np.mean((sample_scaled - sample_reconstructed) ** 2)
        total_mse += mse

    avg_mse = total_mse / n_error_samples

    # 显示重建误差
    ax5.text(0.98, 0.02,
             f'平均重建误差\nMSE = {avg_mse:.6f}',
             transform=ax5.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    print(f"   使用 {n_error_samples} 个样本的平均重建误差: {avg_mse:.6f}")

    plt.tight_layout()
    plt.savefig('visualize_PCA_result2.png', dpi=600)
    plt.show()

    # 4. 添加额外统计信息
    print("\n6. 额外统计信息:")
    print(f"   原始特征维度: {original_dim}")
    print(f"   保留95%方差所需维度: {compressed_dim}")
    print(f"   维度压缩率: {compression_rate:.2f}%")
    print(f"   信息保留率: 95.00%")
    print(f"   平均重建MSE: {avg_mse:.6f}")

    # 返回有用的信息
    return {
        'n_components_95': n_components_95,
        'compression_rate': compression_rate,
        'avg_mse': avg_mse,
        'cumulative_variance': cumulative_variance,
        'individual_variance': individual_variance
    }

def main():
    """
    主函数：执行完整的数据转换和降维流程
    """
    # ==============================
    # 1. 加载数据
    # ==============================
    print("=" * 60)
    print("MNIST数据集转换与降维处理")
    print("=" * 60)

    # 请将以下路径替换为您的实际文件路径 ！！！！！！！
    image_file = 'E:/Download/MNIST/t10k-images-idx3-ubyte.gz'  # 10000张测试图像
    # 如果是训练集，使用: 'train-images-idx3-ubyte.gz'

    try:
        # 加载图像数据
        images = load_idx3_ubyte(image_file)
        # 加载标签文件
        label_file = 'E:/Download/MNIST/t10k-labels-idx1-ubyte.gz'  # 标签文件
        with gzip.open(label_file, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        print(f"   标签形状: {labels.shape}")
        print(f"   标签类型: {labels.dtype}")
        print(f"   标签唯一值: {np.unique(labels)}")
        # ==============================
        # 2. 数据预处理
        # ==============================
        X_scaled, scaler = preprocess_data(images)

        # ==============================
        # 3. PCA降维
        # ==============================
        X_pca, pca = apply_pca(X_scaled, variance_ratio=0.95)

        # ==============================
        # 4. 可视化结果
        # ==============================
        visualize_pca_results(pca,X_scaled, scaler, n_samples=5)

        # ==============================
        # 5. 保存结果（可选）
        # ==============================
        print("\n=== 结果保存 ===")
        # 保存降维后的特征矩阵
        np.save('mnist_features_pca.npy', X_pca)
        print(f"   降维后的特征矩阵已保存为: mnist_features_pca.npy")
        print(f"   形状: {X_pca.shape}")

        # 保存PCA模型（用于后续新数据的转换）
        import joblib
        joblib.dump(pca, 'pca_model_95.pkl')
        print(f"   PCA模型已保存为: pca_model_95.pkl")

        # ==============================
        # 6. 为机器学习建模准备数据
        # ==============================
        print("\n=== 机器学习建模准备 ===")
        print(f"   最终特征矩阵形状: {X_pca.shape}")
        print(f"   可用于监督学习的特征数量: {X_pca.shape[1]}")
        print(f"   样本数量: {X_pca.shape[0]}")

        # 示例：如何加载标签文件（如果需要）
        print("\n注意：如果需要标签进行监督学习，请使用以下代码加载标签:")
        print("""
        label_file = 't10k-labels-idx1-ubyte.gz'  # 标签文件
        with gzip.open(label_file, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        """)

        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n错误: 文件未找到 - {e}")
        print("请确保:")
        print("1. 文件路径正确")
        print("2. 文件存在于指定路径")
        print("3. 文件名为: t10k-images-idx3-ubyte.gz")

    except Exception as e:
        print(f"\n错误: {e}")
        print("请检查文件格式是否正确")

if __name__ == "__main__":
    main()