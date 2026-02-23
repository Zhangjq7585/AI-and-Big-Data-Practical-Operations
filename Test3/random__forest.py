##############
'''
    方法2：基于模型的特征重要性选择
优点：
    考虑了特征与目标的关系
    可以捕捉非线性关系
    提供特征重要性排序
    通常能保持或提高模型性能
缺点：
    计算成本较高
    需要训练模型
    结果依赖于所选模型
    可能选择高度相关的特征
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
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

# 分割数据集
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 方法2：基于随机森林的特征重要性选择

# 训练随机森林模型获取特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_full, y_train)

# 获取特征重要性
feature_importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 可视化特征重要性
plt.figure(figsize=(12, 6))
plt.bar(range(20), feature_importance_df['importance'].head(20))
plt.xticks(range(20), feature_importance_df['feature'].head(20), rotation=90)
plt.xlabel('特征')
plt.ylabel('重要性')
plt.title('Top 20 特征重要性')
plt.tight_layout()
plt.savefig('Top-20特征重要性.png', dpi=600)
plt.show()

# 使用SelectFromModel选择重要特征
# 方法2.1：基于阈值选择
selector_threshold = SelectFromModel(rf, threshold='median', prefit=True)
X_train_selected_th = selector_threshold.transform(X_train_full)
X_test_selected_th = selector_threshold.transform(X_test_full)

# 获取选择的特征
selected_features_th = X.columns[selector_threshold.get_support()]

print(f"\n基于阈值选择后的特征数量: {X_train_selected_th.shape[1]}")
print(f"选择的特征: {list(selected_features_th)}")

# 方法2.2：基于top k特征选择
# 选择重要性最高的k个特征
k = 50  # 选择50个最重要的特征
selector_topk = SelectFromModel(rf, max_features=k, prefit=True)
X_train_selected_k = selector_topk.transform(X_train_full)
X_test_selected_k = selector_topk.transform(X_test_full)

# 获取选择的特征
selected_features_k = X.columns[selector_topk.get_support()]

print(f"\n基于top-{k}选择后的特征数量: {X_train_selected_k.shape[1]}")
print(f"选择的特征: {list(selected_features_k)}")

# 评估特征选择效果
print("\n评估特征选择效果:")
print("-" * 50)

# 使用所有特征训练模型
rf_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_full.fit(X_train_full, y_train)
y_pred_full = rf_full.predict(X_test_full)
accuracy_full = accuracy_score(y_test, y_pred_full)

# 使用基于阈值选择的特征训练模型
rf_th = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_th.fit(X_train_selected_th, y_train)
y_pred_th = rf_th.predict(X_test_selected_th)
accuracy_th = accuracy_score(y_test, y_pred_th)

# 使用top-k选择的特征训练模型
rf_k = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_k.fit(X_train_selected_k, y_train)
y_pred_k = rf_k.predict(X_test_selected_k)
accuracy_k = accuracy_score(y_test, y_pred_k)

print(f"使用所有特征 ({X_train_full.shape[1]}) 的准确率: {accuracy_full:.4f}")
print(f"使用阈值选择特征 ({X_train_selected_th.shape[1]}) 的准确率: {accuracy_th:.4f}")
print(f"使用top-{k}选择特征 ({X_train_selected_k.shape[1]}) 的准确率: {accuracy_k:.4f}")

# 计算特征减少比例和准确率保持情况
reduction_th = 1 - X_train_selected_th.shape[1] / X_train_full.shape[1]
reduction_k = 1 - X_train_selected_k.shape[1] / X_train_full.shape[1]

print(f"\n特征减少比例:")
print(f"阈值选择法: {reduction_th:.2%}")
print(f"Top-{k}选择法: {reduction_k:.2%}")

print(f"\n准确率保持比例:")
print(f"阈值选择法: {accuracy_th/accuracy_full:.2%}")
print(f"Top-{k}选择法: {accuracy_k/accuracy_full:.2%}")


