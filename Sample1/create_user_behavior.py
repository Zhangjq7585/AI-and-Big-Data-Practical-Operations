# 将电商用户数据集扩充为500个
'''
列名                 类型      说明
user_id             string    用户ID (USER_0001-USER_0500)
age                 float     年龄 (18-60岁，15%缺失值)
gender              int       性别 (1:男, 0:女)
income              float     月收入 (3-30千元)
browsing_minutes    float     浏览时长 (5-180分钟)
pages_viewed        int       浏览页面数 (3-100页)
click_rate          float     点击率 (0.05-0.8)
previous_purchase   int       历史是否购买 (0/1)
purchase            int       是否购买 (目标变量, 0/1)
'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 设置随机种子以保证可复现性
np.random.seed(42)
random.seed(42)

# 生成500个用户的扩展数据
n_users = 500
user_ids = [f'USER_{i:04d}' for i in range(1, n_users + 1)]

# 生成用户基本信息
print("正在生成用户基本信息...")

# 年龄：18-60岁，正态分布，添加缺失值
ages = []
for i in range(n_users):
    if random.random() < 0.15:  # 15%缺失值
        ages.append(np.nan)
    else:
        age = int(np.random.normal(loc=32, scale=8))
        age = max(18, min(age, 60))
        ages.append(age)

# 性别：略微女性偏多（电商常见现象）
genders = np.random.choice(['男', '女'], size=n_users, p=[0.45, 0.55])

# 生成消费行为数据
print("正在生成消费行为数据...")

all_records = []
base_date = datetime(2024, 1, 1)

# 不同用户群体的消费模式
user_segments = ['低消费低频', '低消费高频', '高消费低频', '高消费高频', '异常用户']
segment_weights = [0.25, 0.35, 0.25, 0.10, 0.05]  # 不同用户群体的比例

for i, user_id in enumerate(user_ids):
    # 为用户分配消费模式
    segment = np.random.choice(user_segments, p=segment_weights)

    # 根据用户群体确定消费次数
    if segment == '低消费低频':
        n_purchases = random.randint(1, 3)
        purchase_amount_mean = 50
        purchase_amount_std = 20
    elif segment == '低消费高频':
        n_purchases = random.randint(8, 15)
        purchase_amount_mean = 80
        purchase_amount_std = 30
    elif segment == '高消费低频':
        n_purchases = random.randint(2, 5)
        purchase_amount_mean = 500
        purchase_amount_std = 200
    elif segment == '高消费高频':
        n_purchases = random.randint(10, 20)
        purchase_amount_mean = 300
        purchase_amount_std = 100
    else:  # 异常用户
        n_purchases = random.randint(1, 5)
        purchase_amount_mean = 1500
        purchase_amount_std = 1000

    # 生成历史购买记录（影响当前购买行为）
    historical_purchase_rate = random.random()  # 0-1之间的值

    for j in range(n_purchases):
        # 基础消费金额（根据用户群体）
        base_amount = np.random.normal(loc=purchase_amount_mean, scale=purchase_amount_std)

        # 添加季节性波动（周末、节假日消费更高）
        days_offset = random.randint(0, 365)
        purchase_date = base_date + timedelta(days=days_offset)
        is_weekend = purchase_date.weekday() >= 5
        is_holiday = purchase_date.month in [1, 2, 6, 10, 12]  # 节假日月份

        # 金额调整因子
        amount_factor = 1.0
        if is_weekend:
            amount_factor *= random.uniform(1.1, 1.3)
        if is_holiday:
            amount_factor *= random.uniform(1.2, 1.5)

        # 添加少量异常值（3%的概率）
        if random.random() < 0.03:
            base_amount *= random.uniform(5, 20)  # 异常值放大5-20倍

        # 最终金额计算
        final_amount = max(10, base_amount * amount_factor)  # 最小10元

        # 浏览时长：与消费金额正相关
        browsing_minutes = np.random.normal(loc=30 + final_amount / 20, scale=15)
        browsing_minutes = max(5, min(browsing_minutes, 180))

        # 浏览页面数：与浏览时长相关
        pages_viewed = int(browsing_minutes * np.random.uniform(0.2, 0.5))
        pages_viewed = max(3, min(pages_viewed, 100))

        # 点击率：0.05-0.8之间，与购买概率正相关
        click_rate = random.uniform(0.05, 0.8)

        # 历史是否购买：基于历史购买率
        previous_purchase = 1 if random.random() < historical_purchase_rate else 0

        # 是否购买（目标变量）的计算逻辑
        # 购买概率受多个因素影响
        purchase_probability = 0.3  # 基础概率

        # 影响因素权重
        if browsing_minutes > 60:
            purchase_probability += 0.2
        if pages_viewed > 30:
            purchase_probability += 0.1
        if click_rate > 0.5:
            purchase_probability += 0.15
        if previous_purchase == 1:
            purchase_probability += 0.15
        if final_amount > 200:  # 高价值商品更可能购买
            purchase_probability += 0.1
        if 25 <= ages[i] <= 40:  # 主力消费年龄段
            purchase_probability += 0.1

        # 添加随机噪声
        purchase_probability += random.uniform(-0.1, 0.1)
        purchase_probability = min(max(purchase_probability, 0.05), 0.95)

        # 决定是否购买
        purchase = 1 if random.random() < purchase_probability else 0

        all_records.append({
            'user_id': user_id,
            'age': ages[i],
            'gender': 1 if genders[i] == '男' else 0,  # 1:男, 0:女
            'income': np.random.normal(loc=12, scale=5),  # 月收入（千元）
            'browsing_minutes': round(browsing_minutes, 1),
            'pages_viewed': pages_viewed,
            'click_rate': round(click_rate, 3),
            'previous_purchase': previous_purchase,
            'purchase': purchase,
            'purchase_amount': round(final_amount, 2),
            'purchase_date': purchase_date.strftime('%Y-%m-%d'),
            'user_segment': segment
        })

# 创建DataFrame
df = pd.DataFrame(all_records)

# 打乱数据顺序
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存主要数据集（用于建模）
df_main = df[['user_id', 'age', 'gender', 'income', 'browsing_minutes',
              'pages_viewed', 'click_rate', 'previous_purchase', 'purchase']].copy()

# 调整收入范围和格式
df_main['income'] = df_main['income'].apply(lambda x: round(max(3, min(x, 30)), 1))

# 保存完整数据集（用于分析）
df_full = df.copy()

# 保存为CSV文件
df_main.to_csv('user_behavior.csv', index=False)
df_full.to_csv('user_behavior_full.csv', index=False)

print("=" * 60)
print("✅ 数据集生成完成！")
print("=" * 60)
print(f"总样本数：{len(df)} 条记录")
print(f"用户数：{n_users} 人")
print(f"平均每人消费次数：{len(df) / n_users:.2f} 次")
print("\n📊 数据分布统计：")
print(f"年龄缺失值比例：{df['age'].isna().mean():.2%}")
print(f"性别分布：男性 {df['gender'].mean():.2%}")
print(f"购买比例：{df['purchase'].mean():.2%}")

print("\n📈 特征统计信息：")
print(df_main.describe())

print("\n📁 生成的文件：")
print("1. user_behavior.csv - 主要建模数据集（精简版）")
print("2. user_behavior_full.csv - 完整分析数据集（含更多字段）")

print("\n🔍 数据集预览（前10行）：")
print(df_main.head(10))