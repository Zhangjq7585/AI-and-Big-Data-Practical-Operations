## 方法4：使用交叉验证和集成方法

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
###############################################################

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import BaggingClassifier

# 方法4.1: 使用交叉验证进行更可靠的评估
svm_cv = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)

# 获取交叉验证预测
cv_predictions = cross_val_predict(svm_cv, X_train_scaled, y_train, cv=5)

# 训练最终模型
svm_cv.fit(X_train_scaled, y_train)

print("交叉验证评估的SVM:")
print(f"交叉验证预测准确率: {np.mean(cv_predictions == y_train):.4f}")
print(f"测试集准确率: {svm_cv.score(X_test_scaled, y_test):.4f}")

# 方法4.2: 使用Bagging集成方法（降低方差）
bagging_svm = BaggingClassifier(
    base_estimator=SVC(kernel='rbf', C=1, gamma='scale', random_state=42),
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

bagging_svm.fit(X_train_scaled, y_train)

print(f"\nBagging集成SVM:")
print(f"训练集准确率: {bagging_svm.score(X_train_scaled, y_train):.4f}")
print(f"测试集准确率: {bagging_svm.score(X_test_scaled, y_test):.4f}")

# 方法4.3: 结合多种预处理和模型参数的Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3)),  # 保留3个主成分
    ('svm', SVC(kernel='rbf', C=1, gamma='auto', probability=True, random_state=42))
])

pipeline.fit(X_train, y_train)

print(f"\nPipeline (标准化+PCA+SVM):")
print(f"训练集准确率: {pipeline.score(X_train, y_train):.4f}")
print(f"测试集准确率: {pipeline.score(X_test, y_test):.4f}")