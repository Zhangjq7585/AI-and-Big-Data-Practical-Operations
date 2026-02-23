import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 设置随机种子以保证可复现性
np.random.seed(42)
random.seed(42)

# 生成用户数据
n_users = 1000
user_ids = [f'USER_{i:04d}' for i in range(1, n_users + 1)]

# 生成年龄（包含15%缺失值）
ages = []
age_missing_rate = 0.15
for i in range(n_users):
    if random.random() < age_missing_rate:
        ages.append(np.nan)
    else:
        # 年龄分布：18-65岁，正态分布
        age = int(np.random.normal(loc=35, scale=10))
        age = max(18, min(age, 65))
        ages.append(age)

# 生成性别
genders = np.random.choice(['男', '女'], size=n_users, p=[0.48, 0.52])

# 生成消费记录（每人1-10条消费记录）
all_records = []
base_date = datetime(2024, 1, 1)

for i, user_id in enumerate(user_ids):
    n_purchases = random.randint(1, 10)

    for j in range(n_purchases):
        # 正常消费金额：10-2000元，对数正态分布
        if random.random() < 0.03:  # 3%的概率生成异常值
            amount = random.uniform(10000, 50000)  # 异常值：10000-50000元
        else:
            amount = np.random.lognormal(mean=5, sigma=1)
            amount = min(max(amount, 10), 2000)  # 限制在10-2000元

        # 消费时间：2024年随机日期
        days_offset = random.randint(0, 365)
        purchase_date = base_date + timedelta(days=days_offset)

        all_records.append({
            '用户ID': user_id,
            '消费金额': round(amount, 2),
            '消费时间': purchase_date.strftime('%Y-%m-%d %H:%M:%S'),
            '年龄': ages[i],
            '性别': genders[i]
        })

# 创建DataFrame
df = pd.DataFrame(all_records)

# 打乱数据顺序
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存为CSV文件
df.to_csv('Dataset/ecommerce_consume.csv', index=False, encoding='utf-8-sig')

print(f"数据集已生成，共 {len(df)} 条记录")
print(f"字段：{list(df.columns)}")
print("\n数据预览：")
print(df.head())
print("\n数据统计信息：")
print(df.describe())
print(f"\n年龄缺失值比例：{df['年龄'].isna().mean():.2%}")
print(f"消费金额异常检查（>5000元）：{(df['消费金额'] > 5000).sum()} 条")