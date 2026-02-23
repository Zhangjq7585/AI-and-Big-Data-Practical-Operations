import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 创建包含100个特征的数据集
# 设置n_informative=50，表示只有50个特征是有信息量的
# 设置n_redundant=30，表示有30个冗余特征（与其他特征相关）
# 设置n_repeated=10，表示有10个重复特征（完全复制其他特征）
# 剩余10个特征是随机噪声
X, y = make_classification(
    n_samples=1000,
    n_features=100,
    n_informative=50,
    n_redundant=30,
    n_repeated=10,
    n_clusters_per_class=2,
    random_state=42
)

# 创建特征名称
feature_names = [f'feature_{i}' for i in range(X.shape[1])]

# 创建DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 保存为CSV文件
df.to_csv('classification_dataset.csv', index=False)

print("数据集已保存为 classification_dataset.csv")
print(f"数据集形状: {df.shape}")
print(f"特征数量: {len(feature_names)}")
print(f"样本数量: {len(df)}")

# 显示前5行数据
print("\n数据集前5行:")
print(df.head())

