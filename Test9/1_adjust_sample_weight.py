# 方法1_adjust_sample_weight.py
"""
方法1：调整样本权重
通过为不同类别分配不同权重来处理不平衡问题
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """加载和准备数据"""
    # 假设数据集已保存为CSV文件
    # 如果未保存，请使用Create_data.py中创建数据集的方法
    try:
        df = pd.read_csv('credit_card_fraud_data.csv')
    except FileNotFoundError:
        print("请先运行Create_data.py或确保数据集文件存在")

    print(f"数据集大小: {df.shape}")
    print(f"欺诈样本比例: {df['is_fraud'].mean():.2%}")

    return df

def method_adjust_sample_weight():
    """调整样本权重方法"""
    print("=" * 60)
    print("方法1: 调整样本权重")
    print("=" * 60)

    # 加载数据
    df = load_and_prepare_data()

    # 划分特征和目标变量
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    print(f"训练集中欺诈比例: {y_train.mean():.2%}")

    # 方法1.1: 使用scale_pos_weight参数 -------直接设置正负样本权重比
    print("\n--- 方法1.1: 使用scale_pos_weight参数 ---")
    neg_pos_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    print(f"负样本/正样本比例: {neg_pos_ratio:.2f}")

    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)

    # 参数设置
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'scale_pos_weight': neg_pos_ratio,  # 关键参数
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # 预测和评估
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['正常', '欺诈']))

    # 方法1.2: 使用自定义权重-------- 自定义权重：为每个样本分配不同权重
    print("\n--- 方法1.2: 使用自定义权重 ---")

    # 计算自定义权重
    class_weight = len(y_train) / (2 * np.bincount(y_train.astype(int)))
    sample_weights = np.where(y_train == 1, class_weight[1], class_weight[0])
    print(f"正样本权重: {class_weight[1]:.2f}, 负样本权重: {class_weight[0]:.2f}")

    # 创建带权重的数据集
    train_data_weighted = lgb.Dataset(X_train, label=y_train, weight=sample_weights)

    params_no_weight = params.copy()
    params_no_weight.pop('scale_pos_weight', None)

    model_weighted = lgb.train(
        params_no_weight,
        train_data_weighted,
        num_boost_round=500,
        valid_sets=[train_data_weighted],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # 预测和评估
    y_pred_proba_weighted = model_weighted.predict(X_test)
    y_pred_weighted = (y_pred_proba_weighted > 0.5).astype(int)

    print(f"AUC: {roc_auc_score(y_test, y_pred_proba_weighted):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_weighted):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_weighted, target_names=['正常', '欺诈']))

    # 方法1.3: 使用is_unbalance参数------让LightGBM自动处理不平衡
    print("\n--- 方法1.3: 使用is_unbalance参数 ---")

    params_unbalance = params_no_weight.copy()
    params_unbalance['is_unbalance'] = True

    model_unbalance = lgb.train(
        params_unbalance,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # 预测和评估
    y_pred_proba_unbalance = model_unbalance.predict(X_test)
    y_pred_unbalance = (y_pred_proba_unbalance > 0.5).astype(int)

    print(f"AUC: {roc_auc_score(y_test, y_pred_proba_unbalance):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_unbalance):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_unbalance, target_names=['正常', '欺诈']))

    return {
        'scale_pos_weight': (model, y_pred_proba, y_pred),
        'custom_weight': (model_weighted, y_pred_proba_weighted, y_pred_weighted),
        'is_unbalance': (model_unbalance, y_pred_proba_unbalance, y_pred_unbalance)
    }

if __name__ == "__main__":
    results = method_adjust_sample_weight()