### 使用numpy创建模拟数据集，生成随机灰度图像数据
'''
    数据转换步骤：
    数据加载与整形：将10000张28×28的灰度图像展平为784维特征向量
    标准化处理：对像素值进行标准化，消除量纲影响
    创建特征矩阵：形成10000×784的特征矩阵

    降维方法：
    使用主成分分析（PCA），这是一种线性降维方法：
    通过正交变换将相关变量转为不相关的主成分
    按方差贡献率选择主成分，保留95%的原始信息
    可以有效减少特征维度，去除噪声和冗余信息
'''

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置随机种子确保可重复性
np.random.seed(42)

def create_and_process_dataset():
    """
    创建数据集并进行转换和降维处理
    """

    # 1. 创建模拟数据集（10000张28×28灰度图）
    print("1. 创建数据集...")
    num_images = 10000
    height, width = 28, 28
    pixel_range = 256  # 灰度值范围0-255

    # 生成随机灰度图像数据（模拟真实图像分布）
    images = np.random.randint(0, pixel_range, size=(num_images, height, width))

    # 添加一些简单的模式，使数据更接近真实图像
    for i in range(num_images):
        # 随机添加一些简单的形状模式
        if i % 5 == 0:
            # 添加垂直线条
            col = np.random.randint(0, width)
            images[i, :, col] = np.random.randint(200, 256)
        elif i % 5 == 1:
            # 添加水平线条
            row = np.random.randint(0, height)
            images[i, row, :] = np.random.randint(200, 256)

    print(f"原始数据形状: {images.shape}")
    print(f"像素值范围: [{images.min()}, {images.max()}]")

    # 2. 保存原始数据集
    print("\n2. 保存数据集...")
    np.save('image_dataset.npy', images)
    print("数据集已保存为 'image_dataset.npy'")

    # 3. 数据转换：展平为特征矩阵
    print("\n3. 数据转换：展平图像...")
    # 将每张28×28的图像展平为784维向量
    X_flattened = images.reshape(num_images, -1)
    print(f"展平后形状: {X_flattened.shape}")

    # 4. 数据标准化
    print("\n4. 数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flattened)
    print(f"标准化后 - 均值: {X_scaled.mean():.4f}, 标准差: {X_scaled.std():.4f}")

    # 5. 降维处理（PCA保留95%信息）
    print("\n5. 降维处理...")
    pca = PCA(n_components=0.95, random_state=42)  # 保留95%方差
    X_pca = pca.fit_transform(X_scaled)

    print(f"降维后形状: {X_pca.shape}")
    print(f"保留的主成分数量: {pca.n_components_}")
    print(f"累计方差解释率: {pca.explained_variance_ratio_.sum():.4f}")

    # 6. 保存处理后的数据
    print("\n6. 保存处理结果...")
    np.save('X_scaled.npy', X_scaled)
    np.save('X_pca.npy', X_pca)
    np.save('pca_components.npy', pca.components_)
    np.save('pca_mean.npy', pca.mean_)

    print("处理后的数据已保存:")
    print("  - X_scaled.npy: 标准化后的特征矩阵")
    print("  - X_pca.npy: 降维后的特征矩阵")
    print("  - pca_components.npy: PCA主成分")
    print("  - pca_mean.npy: PCA均值")

    return images, X_scaled, X_pca, pca


def visualize_results(images, X_pca, pca):
    """
    可视化处理结果
    """
    # 可视化原始图像和重建图像
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # 显示原始图像
    for i in range(5):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title(f'原始图像 {i + 1}')
        axes[0, i].axis('off')

    # 使用PCA重建图像
    X_reconstructed = pca.inverse_transform(X_pca[:5])  # 重建前5个样本
    X_reconstructed = X_reconstructed.reshape(-1, 28, 28)

    for i in range(5):
        axes[1, i].imshow(X_reconstructed[i], cmap='gray')
        axes[1, i].set_title(f'重建图像 {i + 1}')
        axes[1, i].axis('off')

    plt.suptitle('原始图像 vs PCA重建图像', fontsize=16)
    plt.tight_layout()
    plt.savefig('pca_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 绘制方差解释率曲线
    plt.figure(figsize=(10, 6))
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             'b-', linewidth=2, label='累计方差解释率')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
    plt.axvline(x=pca.n_components_, color='g', linestyle='--',
                label=f'主成分数: {pca.n_components_}')

    plt.xlabel('主成分数量', fontsize=12)
    plt.ylabel('方差解释率', fontsize=12)
    plt.title('PCA方差解释率曲线', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pca_variance_explained.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n可视化结果已保存为 'pca_reconstruction.png' 和 'pca_variance_explained.png'")


def load_and_use_dataset():
    """
    加载并使用处理后的数据集
    """
    print("\n加载处理后的数据...")
    try:
        # 加载处理后的数据
        X_scaled = np.load('X_scaled.npy')
        X_pca = np.load('X_pca.npy')

        print(f"加载成功！")
        print(f"标准化特征矩阵形状: {X_scaled.shape}")
        print(f"降维后特征矩阵形状: {X_pca.shape}")

        return X_scaled, X_pca
    except FileNotFoundError:
        print("未找到处理后的数据文件，请先运行数据处理流程。")
        return None, None


def main():
    """
    主函数：执行完整的数据处理流程
    """
    print("=" * 60)
    print("图像数据集转换与降维处理")
    print("=" * 60)

    # 创建并处理数据集
    images, X_scaled, X_pca, pca = create_and_process_dataset()

    print("\n" + "=" * 60)
    print("数据转换与降维完成！")
    print("=" * 60)

    # 显示降维效果统计
    print(f"\n降维效果统计:")
    print(f"原始特征维度: {X_scaled.shape[1]}")
    print(f"降维后特征维度: {X_pca.shape[1]}")
    print(f"维度减少比例: {1 - X_pca.shape[1] / X_scaled.shape[1]:.2%}")
    print(f"信息保留比例: {pca.explained_variance_ratio_.sum():.4f}")

    # 可视化结果
    visualize = input("\n是否可视化处理结果？(y/n): ").lower()
    if visualize == 'y':
        visualize_results(images, X_pca, pca)

    # 展示如何使用处理后的数据
    print("\n" + "=" * 60)
    print("如何使用处理后的数据进行机器学习建模:")
    print("=" * 60)

    # 加载处理后的数据示例
    X_scaled_loaded, X_pca_loaded = load_and_use_dataset()
'''
    if X_pca_loaded is not None:
        print("\n降维后的数据可以直接用于机器学习模型:")
        print("示例代码:")
        print("""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 假设我们有标签 y（这里用随机标签作为示例）
y = np.random.randint(0, 10, size=len(X_pca_loaded))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_pca_loaded, y, test_size=0.2, random_state=42
)

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")
        """)
'''

if __name__ == "__main__":
    main()

