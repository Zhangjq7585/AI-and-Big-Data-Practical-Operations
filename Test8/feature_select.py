## 方法3：特征选择。删除冗余或噪声特征、降低模型输入维度。

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("数据集形状:", X.shape)
print("特征名称:", feature_names)
print("类别名称:", target_names)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print(f"\n训练集样本数: {X_train.shape[0]}")
# print(f"测试集样本数: {X_test.shape[0]}")
# print(f"类别分布 - 训练集: {np.bincount(y_train)}")
# print(f"类别分布 - 测试集: {np.bincount(y_test)}")
################################################################

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# 使用matplotlib内置的字体（如黑体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 'FangSong', 'KaiTi', 'SimSun' 等
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 方法3.1: 使用PCA降维
pca = PCA(n_components=2)  # 降到2维
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

svm_pca = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_pca.fit(X_train_pca, y_train)

print("PCA降维+SVM:")
print(f"解释方差比例: {pca.explained_variance_ratio_.sum():.4f}")
print(f"训练集准确率: {svm_pca.score(X_train_pca, y_train):.4f}")
print(f"测试集准确率: {svm_pca.score(X_test_pca, y_test):.4f}")

# 可视化PCA降维后的数据分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for i, name in enumerate(target_names):
    plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1],
                label=name, alpha=0.7, s=50)
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.title('训练集PCA降维可视化')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for i, name in enumerate(target_names):
    plt.scatter(X_test_pca[y_test == i, 0], X_test_pca[y_test == i, 1],
                label=name, alpha=0.7, s=50)
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.title('测试集PCA降维可视化')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('可视化PCA降维后的数据分布.png',dpi=600)
plt.show()

# 方法3.2: 特征选择
selector = SelectKBest(f_classif, k=2)  # 选择最重要的2个特征
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

svm_selected = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_selected.fit(X_train_selected, y_train)

print(f"\n特征选择后的SVM:")
print(f"选择的特征数量: 2/4")
print(f"训练集准确率: {svm_selected.score(X_train_selected, y_train):.4f}")
print(f"测试集准确率: {svm_selected.score(X_test_selected, y_test):.4f}")