## 方法2：使用核函数正则化。对RBF核，通过gamma参数控制核函数复杂度(gamma越小，核函数越简单，
## 减少过拟合)；

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

#####################################################################
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# 方法2.1: 使用更简单的线性核
svm_linear = SVC(kernel='linear', C=1, random_state=42)
linear_scores = cross_val_score(svm_linear, X_train_scaled, y_train, cv=5, scoring='accuracy')
svm_linear.fit(X_train_scaled, y_train)

print("线性核SVM:")
print(f"交叉验证准确率: {linear_scores.mean():.4f} (+/- {linear_scores.std() * 2:.4f})")
print(f"训练集准确率: {svm_linear.score(X_train_scaled, y_train):.4f}")
print(f"测试集准确率: {svm_linear.score(X_test_scaled, y_test):.4f}")

# 方法2.2: 调整RBF核的gamma参数
param_grid_gamma = {
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    'C': [0.1, 1, 10]
}

svm_gamma = GridSearchCV(
    SVC(kernel='rbf', random_state=42),
    param_grid_gamma,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

svm_gamma.fit(X_train_scaled, y_train)

print(f"\nRBF核最佳参数: {svm_gamma.best_params_}")
print(f"最佳交叉验证准确率: {svm_gamma.best_score_:.4f}")
print(f"测试集准确率: {svm_gamma.score(X_test_scaled, y_test):.4f}")

# 方法2.3: 使用多项式核（调整degree参数）
svm_poly = SVC(kernel='poly', degree=2, C=1, random_state=42)  # 降低多项式阶数
poly_scores = cross_val_score(svm_poly, X_train_scaled, y_train, cv=5, scoring='accuracy')
svm_poly.fit(X_train_scaled, y_train)

print(f"\n多项式核(degree=2) SVM:")
print(f"交叉验证准确率: {poly_scores.mean():.4f} (+/- {poly_scores.std() * 2:.4f})")
print(f"训练集准确率: {svm_poly.score(X_train_scaled, y_train):.4f}")
print(f"测试集准确率: {svm_poly.score(X_test_scaled, y_test):.4f}")
