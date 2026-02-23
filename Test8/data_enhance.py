# 方法5: 通过数据增强增加训练样本
from sklearn.utils import resample
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

# 由于鸢尾花数据集较小，我们可以通过bootstrap重采样来模拟更多数据
X_train_augmented = X_train_scaled.copy()
y_train_augmented = y_train.copy()

# 对每个类别进行重采样，增加样本数量
for class_label in np.unique(y_train):
    # 获取当前类别的样本
    class_mask = y_train == class_label
    X_class = X_train_scaled[class_mask]
    y_class = y_train[class_mask]

    # 重采样增加50%的样本
    n_samples = len(X_class)
    X_resampled, y_resampled = resample(
        X_class, y_class,
        replace=True,
        n_samples=int(n_samples * 1.5),
        random_state=42
    )

    # 添加到增强数据集
    X_train_augmented = np.vstack([X_train_augmented, X_resampled])
    y_train_augmented = np.concatenate([y_train_augmented, y_resampled])

print(f"原始训练集大小: {len(X_train_scaled)}")
print(f"增强后训练集大小: {len(X_train_augmented)}")

# 使用原始数据训练SVM
svm_based = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_based.fit(X_train_scaled, y_train)

# 使用增强数据训练SVM
svm_augmented = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_augmented.fit(X_train_augmented, y_train_augmented)

print(f"\n使用原始数据的SVM:")
print(f"训练集准确率: {svm_based.score(X_train_scaled, y_train):.4f}")
print(f"测试集准确率: {svm_based.score(X_test_scaled, y_test):.4f}")

print(f"\n使用数据增强的SVM:")
print(f"训练集准确率: {svm_augmented.score(X_train_augmented, y_train_augmented):.4f}")
print(f"测试集准确率: {svm_augmented.score(X_test_scaled, y_test):.4f}")

