## 加载鸢尾花数据集并保存为.csv
import pandas as pd
import numpy as np
from sklearn import datasets

# 加载鸢尾花数据集
iris = datasets.load_iris()

# 创建DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# 添加目标列
iris_df['target'] = iris.target

# 添加品种名称列（可选）
iris_df['species'] = iris.target_names[iris.target]

# 显示数据集信息
print("数据集形状:", iris_df.shape)
print("\n前5行数据:")
print(iris_df.head())
print("\n数据集信息:")
print(iris_df.info())
print("\n类别分布:")
print(iris_df['species'].value_counts())

# 方法1: 保存为CSV文件（不包含索引）
iris_df.to_csv('iris_dataset.csv', index=False)
print("\n已保存为: iris_dataset.csv")

