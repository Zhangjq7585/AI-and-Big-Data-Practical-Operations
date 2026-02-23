###########
'''
方法1：基于相关性的特征选择
优点：
    计算效率高，不需要训练模型
    易于理解和解释
    可以有效去除冗余特征
    无监督方法，不依赖标签

缺点：
    只考虑特征间的线性关系
    可能删除对模型重要的特征
    需要手动设置相关性阈值
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 导入数据集
df = pd.read_csv('classification_dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"原始特征数量: {X.shape[1]}")

# 方法1：基于相关性的特征选择---Feature selection based on correlation
# 计算特征之间的相关性矩阵
correlation_matrix = X_scaled.corr().abs()

# 创建掩码来隐藏上三角矩阵（对称部分）
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 可视化相关性矩阵（可选）
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm',
            center=0, square=True, linewidths=.5)
plt.title('特征相关性矩阵')
plt.tight_layout()
plt.savefig('特征相关性矩阵.png', dpi=600)
plt.show()

# 设置相关性阈值
correlation_threshold = 0.8

# 找到高度相关的特征对
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > correlation_threshold:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print(f"找到 {len(high_corr_pairs)} 对高度相关（> {correlation_threshold}）的特征")


# 移除高度相关的特征
def remove_highly_correlated_features(X, threshold=0.8):
    """
    移除高度相关的特征
    策略：对于每一对高度相关的特征，保留其中一个
    """
    corr_matrix = X.corr().abs()

    # 获取上三角矩阵
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 找到相关性高于阈值的特征
    to_drop = []
    for column in upper_tri.columns:
        # 检查是否有相关性高于阈值的特征
        high_corr_features = upper_tri[column][upper_tri[column] > threshold]
        if not high_corr_features.empty:
            # 找到与当前特征最相关的特征
            max_corr_feature = high_corr_features.idxmax()
            max_corr_value = high_corr_features.max()

            # 计算每个特征与目标的相关性（如果y存在）
            # 这里我们保留与目标相关性更高的特征（如果有y的话）
            # 如果没有目标信息，我们随机保留一个

            # 在实际应用中，可能需要考虑特征的重要性
            # 这里我们简单地保留列名在字母顺序上靠前的特征
            if column < max_corr_feature:
                to_drop.append(max_corr_feature)
            else:
                to_drop.append(column)

    # 去重
    to_drop = list(set(to_drop))

    print(f"将移除 {len(to_drop)} 个高度相关的特征")
    print(f"移除的特征: {to_drop}")

    # 保留不高度相关的特征
    selected_features = [col for col in X.columns if col not in to_drop]

    return X[selected_features], selected_features


# 应用基于相关性的特征选择
X_corr_selected, selected_features_corr = remove_highly_correlated_features(
    X_scaled, threshold=correlation_threshold
)

print(f"\n基于相关性特征选择后的特征数量: {X_corr_selected.shape[1]}")

# 验证选择效果
correlation_matrix_selected = X_corr_selected.corr().abs()
max_corr_selected = correlation_matrix_selected.values[np.triu_indices_from(correlation_matrix_selected, k=1)].max()
print(f"选择后特征之间的最大相关性: {max_corr_selected:.4f}")

# 将选择后的数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_corr_selected, y, test_size=0.2, random_state=42
)

print(f"\n训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")


