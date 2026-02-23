## 创建客户流失预测数据集

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 创建模拟的客户流失数据集
np.random.seed(42)
n_samples = 1000

# 生成数值特征
tenure = np.random.randint(1, 72, n_samples)  # 客户时长（月）1个月到72个月之间
monthly_charges = np.random.uniform(20, 120, n_samples)  # 月消费 20-120元
total_charges = tenure * monthly_charges  # 总消费=客户时长*月消费
age = np.random.randint(18, 80, n_samples)  # 客户年龄 18-80岁
dependents = np.random.randint(0, 5, n_samples)  # 家属数量 0-5人

# 生成分类特征
contract_types = ['Month-to-month', 'One year', 'Two year'] #套餐类型
contract = np.random.choice(contract_types, n_samples, p=[0.5, 0.3, 0.2]) #每种套餐类型所占比例

# 支付方式
payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
payment_method = np.random.choice(payment_methods, n_samples)

internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])

# 生成标签（是否流失） - 添加一些逻辑关系
# 假设流失概率与以下因素相关：
churn_prob = (
    0.3 * (contract == 'Month-to-month') +
    0.2 * (internet_service == 'Fiber optic') +
    0.1 * (monthly_charges > 80) +
    0.05 * (tenure < 12) -
    0.1 * (contract == 'Two year') -
    0.05 * (tenure > 24) +
    np.random.normal(0, 0.2, n_samples)
)
churn = (churn_prob > 0.5).astype(int)

# 创建DataFrame
df = pd.DataFrame({
    'customer_id': range(1, n_samples + 1),
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'age': age,
    'dependents': dependents,
    'contract_type': contract,
    'payment_method': payment_method,
    'internet_service': internet_service,
    'churn': churn
})

print("数据集基本信息：")
print(f"数据集形状：{df.shape}")
print(f"流失比例：{df['churn'].mean():.2%}")
print("\n前5行数据：")
print(df.head(5))
print("\n特征数据类型：")
print(df.dtypes)

df.to_csv('DATA.csv', index=False)  #保存原始数据集
