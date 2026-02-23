## 房价预测数据集

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def create_realistic_house_price_dataset(n_samples=1000, random_state=42):
    ## 创建更真实的房价数据集，确保没有负值
    np.random.seed(random_state)

    # 生成更符合实际的房价特征
    n_samples = 1000

    # 生成特征数据
    house_area = np.random.normal(120, 40, n_samples)  # 房屋面积，均值120平米
    house_area = np.clip(house_area, 50, 300)  # 限制在50-300平米

    rooms = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.3, 0.3, 0.15, 0.05])
    bathrooms = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.5, 0.25, 0.05])
    floors = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    build_year = np.random.randint(1990, 2025, n_samples)
    location_score = np.random.uniform(60, 100, n_samples)  # 地理位置评分
    subway_distance = np.random.exponential(2, n_samples)  # 地铁距离（指数分布）
    subway_distance = np.clip(subway_distance, 0.1, 10)
    school_score = np.random.uniform(70, 100, n_samples)  # 学校评分

    # 创建特征矩阵
    X = np.column_stack([
        house_area,
        rooms,
        bathrooms,
        floors,
        build_year,
        location_score,
        subway_distance,
        school_score
    ])

    feature_names = [
        '房屋面积(平方米)',
        '房间数',
        '浴室数',
        '楼层数',
        '建造年份',
        '地理位置评分',
        '地铁距离(公里)',
        '学校评分'
    ]

    # 生成真实的房价（确保都是正值）
    # 基础房价公式
    base_price = (
            house_area * 10000 +  # 面积基础：1万元/平米
            rooms * 200000 +  # 每间房加20万
            bathrooms * 150000 +  # 每个浴室加15万
            (2025 - build_year) * (-5000) +  # 房龄每年减5000
            location_score * 10000 +  # 位置评分每分加1万
            np.exp(-subway_distance / 2) * 300000 +  # 地铁距离近的溢价
            school_score * 8000  # 学校评分每分加8000
    )

    # 添加随机波动
    noise = np.random.normal(0, 200000, n_samples)  # 20万的随机波动
    y = base_price + noise

    # 确保所有房价为正（理论上应该都是正的，但以防万一）
    y = np.maximum(y, 500000)  # 最低房价50万元

    # 转换为万元单位
    y = y / 10000

    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['房价(万元)'] = y

    df.to_excel("Data.xlsx", index=False)
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    print(f"数据集创建完成！")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"房价范围: {y.min():.2f} - {y.max():.2f} 万元")
    print(f"房价均值: {y.mean():.2f} 万元")
    print(f"房价中位数: {np.median(y):.2f} 万元")

    print("\n特征描述统计:")
    print(df.describe().round(2))

    print("\n前5条数据样本:")
    print(df.head().round(2))

    return X_train, X_test, y_train, y_test, df, feature_names


# 创建修正后的数据集
X_train, X_test, y_train, y_test, df, feature_names = create_realistic_house_price_dataset()

# 可视化房价分布
import matplotlib.pyplot as plt

# 设置全局字体为支持中文的字体，例如使用SimHei（黑体）
plt.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei', 'Arial Unicode MS' 等
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 房价分布直方图
axes[0].hist(df['房价(万元)'], bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(df['房价(万元)'].mean(), color='red', linestyle='--', label=f'均值: {df["房价(万元)"].mean():.1f}万')
axes[0].axvline(df['房价(万元)'].median(), color='black', linestyle='--',
                label=f'中位数: {df["房价(万元)"].median():.1f}万')
axes[0].set_xlabel('房价(万元)')
axes[0].set_ylabel('频数')
axes[0].set_title('房价分布')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 房价与面积关系
axes[1].scatter(df['房屋面积(平方米)'], df['房价(万元)'], alpha=0.6)
axes[1].set_xlabel('房屋面积(平方米)')
axes[1].set_ylabel('房价(万元)')
axes[1].set_title('房价 vs 房屋面积')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('房价分布可视化.png', dpi=600)
plt.show()

# 检查是否有负值
print("\n数据检查:")
print(f"是否有负的房价值: {(df['房价(万元)'] < 0).any()}")
print(f"最小房价值: {df['房价(万元)'].min():.2f} 万元")
print(f"负值数量: {(df['房价(万元)'] < 0).sum()}")