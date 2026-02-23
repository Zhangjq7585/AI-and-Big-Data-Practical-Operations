
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 创建模拟的信用卡欺诈检测数据集 ====================
def create_credit_card_fraud_dataset(n_samples=100000, fraud_ratio=0.01):
    """
    创建模拟的信用卡欺诈检测数据集
    特征包括：交易金额、时间、商户类别、地理位置等
    """
    np.random.seed(42)

    # 正常样本数
    n_normal = int(n_samples * (1 - fraud_ratio))
    # 欺诈样本数
    n_fraud = n_samples - n_normal

    # 为正常交易生成数据
    normal_amount = np.random.exponential(100, n_normal)    # 用对数正态分布模拟
    normal_time = np.random.uniform(0, 24, n_normal)  # 24小时制
    # 商户类别
    normal_merchant = np.random.choice(range(1, 11), n_normal, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05, 0.05, 0.05])
    normal_geo = np.random.choice(range(1, 21), n_normal)   # 地理位置
    normal_labels = np.zeros(n_normal)

    # 为欺诈交易生成数据（模式有所不同）
    fraud_amount = np.random.exponential(500, n_fraud)  # 欺诈交易金额更大
    fraud_time = np.random.choice([0, 1, 2, 3, 4, 21, 22, 23], n_fraud)  # 深夜交易更多。21时-次日4时
    fraud_merchant = np.random.choice([1, 5, 8, 9, 10], n_fraud, p=[0.3, 0.25, 0.2, 0.15, 0.1])  # 特定商户欺诈率高
    fraud_geo = np.random.choice([3, 7, 12, 15, 19], n_fraud)  # 特定地区欺诈率高
    fraud_labels = np.ones(n_fraud)

    # 合并数据
    amounts = np.concatenate([normal_amount, fraud_amount])
    times = np.concatenate([normal_time, fraud_time])
    merchants = np.concatenate([normal_merchant, fraud_merchant])
    geos = np.concatenate([normal_geo, fraud_geo])
    labels = np.concatenate([normal_labels, fraud_labels])

    # 添加一些噪声和额外特征
    user_history = np.random.randint(1, 100, n_samples)
    device_type = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 0: 网页, 1: 移动端

    # 创建DataFrame
    df = pd.DataFrame({
        'amount': amounts,
        'time': times,
        'merchant_category': merchants,
        'geo_location': geos,
        'user_history': user_history,
        'device_type': device_type,
        'is_fraud': labels
    })

    # 添加一些特征交互
    df['amount_time_interaction'] = df['amount'] * (df['time'] > 22).astype(int)
    df['amount_merchant_risk'] = df['amount'] * (df['merchant_category'].isin([1, 5, 8, 9, 10])).astype(int)

    # 打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv('credit_card_fraud_data.csv', index=False)    #保存数据集
    return df

# 创建数据集
print("创建模拟信用卡欺诈检测数据集...")
credit_data = create_credit_card_fraud_dataset()
print(f"数据集大小: {credit_data.shape}")
print(f"欺诈样本比例: {credit_data['is_fraud'].mean():.2%}")
print(f"正常样本数: {len(credit_data[credit_data['is_fraud'] == 0])}")
print(f"欺诈样本数: {len(credit_data[credit_data['is_fraud'] == 1])}")

# 划分特征和目标变量
X = credit_data.drop('is_fraud', axis=1)
y = credit_data['is_fraud']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"训练集中欺诈比例: {y_train.mean():.2%}")
print(f"测试集中欺诈比例: {y_test.mean():.2%}")