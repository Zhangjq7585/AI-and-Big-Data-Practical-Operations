## 方法1：增加正则化强度。SVM中通过C参数控制正则化（C越小，正则化越强，避免过拟合）；

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

print(f"\n训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"类别分布 - 训练集: {np.bincount(y_train)}")
print(f"类别分布 - 测试集: {np.bincount(y_test)}")

#######################################################
## 调整正则化参数C
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt

# 使用matplotlib内置的字体（如黑体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 'FangSong', 'KaiTi', 'SimSun' 等
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 创建基础SVM模型（容易过拟合的配置）
svm_base = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_base.fit(X_train_scaled, y_train)

print("基础SVM模型:")
print(f"训练集准确率: {svm_base.score(X_train_scaled, y_train):.4f}")
print(f"测试集准确率: {svm_base.score(X_test_scaled, y_test):.4f}")

# 使用网格搜索找到合适的C值
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear']
}

svm_grid = GridSearchCV(
    SVC(random_state=42, gamma='scale'),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

svm_grid.fit(X_train_scaled, y_train)

print(f"\n最佳参数: {svm_grid.best_params_}")
print(f"最佳交叉验证准确率: {svm_grid.best_score_:.4f}")
print(f"测试集准确率: {svm_grid.score(X_test_scaled, y_test):.4f}")

# 可视化不同C值的影响
C_values = [0.01, 0.1, 1, 10, 100]
train_scores = []
test_scores = []

for C in C_values:
    svm = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    train_scores.append(svm.score(X_train_scaled, y_train))
    test_scores.append(svm.score(X_test_scaled, y_test))

plt.figure(figsize=(10, 6))
plt.plot(C_values, train_scores, 'o-', label='训练集准确率')
plt.plot(C_values, test_scores, 's-', label='测试集准确率')
plt.xscale('log')
plt.xlabel('正则化参数C (log scale)')
plt.ylabel('准确率')
plt.title('不同C值对SVM性能的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('不同C值对SVM性能的影响.png',dpi=600)
plt.show()