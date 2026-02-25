import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## 一、数据质量检查 ##
# 读取数据
df = pd.read_csv('user_behavior.csv')

print("=" * 60)
print("📋 数据质量检查报告")
print("=" * 60)

# 1. 基本统计
print("1. 数据集基本信息：")
print(f"  样本数：{len(df)}")
print(f"  特征数：{len(df.columns)}")
print(f"  内存使用：{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# 2. 缺失值检查
print("\n2. 缺失值检查：")
missing_info = df.isnull().sum()
for col, count in missing_info.items():
    if count > 0:
        print(f"  {col}: {count}个缺失值 ({count/len(df)*100:.1f}%)")

# 3. 数据类型检查
print("\n3. 数据类型：")
print(df.dtypes)

# 4. 目标变量分布
print("\n4. 目标变量分布：")
purchase_dist = df['purchase'].value_counts(normalize=True)
print(f"  购买 (1): {purchase_dist[1]:.2%}")
print(f"  未购买 (0): {purchase_dist[0]:.2%}")

# 5. 异常值检测
print("\n5. 异常值检测（基于IQR）：")
for col in ['age', 'income', 'browsing_minutes', 'pages_viewed']:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"  {col}: {len(outliers)}个异常值 ({len(outliers)/len(df)*100:.1f}%)")

# 6. 相关性分析
print("\n6. 特征与目标变量的相关性：")
correlations = df.corr()['purchase'].sort_values(ascending=False)
for feature, corr in correlations.items():
    if feature != 'purchase':
        print(f"  {feature}: {corr:.3f}")

print("\n" + "=" * 60)
print("✅ 数据质量检查完成！")

## 二、数据可视化分析 ##
# 创建可视化图表
plt.rcParams['font.sans-serif'] = ['SimHei']    #防止出现中文乱码
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('用户行为数据分布分析', fontsize=16)

# 1. 年龄分布
axes[0, 0].hist(df['age'].dropna(), bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('年龄分布')
axes[0, 0].set_xlabel('年龄')
axes[0, 0].set_ylabel('频数')

# 2. 收入分布
axes[0, 1].hist(df['income'], bins=20, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_title('收入分布')
axes[0, 1].set_xlabel('月收入（千元）')

# 3. 浏览时长 vs 购买
axes[0, 2].boxplot([df[df['purchase']==0]['browsing_minutes'],
                    df[df['purchase']==1]['browsing_minutes']],
                   labels=['未购买', '购买'])
axes[0, 2].set_title('浏览时长 vs 购买行为')
axes[0, 2].set_ylabel('浏览时长（分钟）')

# 4. 点击率分布
axes[1, 0].hist(df['click_rate'], bins=20, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_title('点击率分布')
axes[1, 0].set_xlabel('点击率')

# 5. 性别购买比例
gender_purchase = df.groupby('gender')['purchase'].mean()
axes[1, 1].bar(['女', '男'], gender_purchase.values, color=['pink', 'lightblue'])
axes[1, 1].set_title('性别购买比例')
axes[1, 1].set_ylabel('购买比例')

# 6. 相关性热图
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[1, 2])
axes[1, 2].set_title('特征相关性热图')

plt.tight_layout()
plt.savefig('data_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("📈 可视化图表已保存为: data_distribution_analysis.png")