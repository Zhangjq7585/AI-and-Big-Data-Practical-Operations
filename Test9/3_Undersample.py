
# method3_undersampling.py
"""
方法3：欠采样方法
通过减少多数类样本来平衡数据集
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载数据"""
    try:
        df = pd.read_csv('credit_card_fraud_data.csv')
    except FileNotFoundError:
        print("请先运行method1或确保数据集文件存在")
        return None
    return df

def method_undersampling():
    """欠采样方法"""
    print("=" * 60)
    print("方法3: 欠采样方法")
    print("=" * 60)

    # 加载数据
    df = load_data()
    if df is None:
        return

    # 划分特征和目标变量
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print(f"原始训练集大小: {X_train.shape}, 欺诈比例: {y_train.mean():.2%}")

    # 方法3.1: 随机欠采样----随机删除多数类样本
    print("\n--- 方法3.1: 随机欠采样 ---")

    # 分离多数类和少数类
    X_train_majority = X_train[y_train == 0]
    X_train_minority = X_train[y_train == 1]
    y_train_majority = y_train[y_train == 0]
    y_train_minority = y_train[y_train == 1]

    # 欠采样多数类
    from sklearn.utils import resample

    # 将多数类样本数量减少到与少数类成一定比例
    # 这里设置为少数类的5倍（即5:1的比例）
    n_samples = min(len(X_train_minority) * 5, len(X_train_majority))

    X_train_majority_downsampled, y_train_majority_downsampled = resample(
        X_train_majority,
        y_train_majority,
        replace=False,  # 不放回采样
        n_samples=n_samples,
        random_state=42
    )

    # 合并欠采样后的数据
    X_train_downsampled = pd.concat([X_train_majority_downsampled, X_train_minority])
    y_train_downsampled = pd.concat([y_train_majority_downsampled, y_train_minority])

    # 打乱数据
    shuffle_idx = np.random.permutation(len(X_train_downsampled))
    X_train_downsampled = X_train_downsampled.iloc[shuffle_idx].reset_index(drop=True)
    y_train_downsampled = y_train_downsampled.iloc[shuffle_idx].reset_index(drop=True)

    print(f"欠采样后训练集大小: {X_train_downsampled.shape}, 欺诈比例: {y_train_downsampled.mean():.2%}")

    # 训练模型
    train_data = lgb.Dataset(X_train_downsampled, label=y_train_downsampled)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    model_downsampled = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # 预测和评估
    y_pred_proba_downsampled = model_downsampled.predict(X_test)
    y_pred_downsampled = (y_pred_proba_downsampled > 0.5).astype(int)

    print(f"AUC: {roc_auc_score(y_test, y_pred_proba_downsampled):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_downsampled):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_downsampled, target_names=['正常', '欺诈']))

    # 方法3.2: NearMiss欠采样----选择有代表性的多数类样本
    print("\n--- 方法3.2: NearMiss欠采样 ---")

    try:
        from imblearn.under_sampling import NearMiss

        # 应用NearMiss
        near_miss = NearMiss(version=3, n_neighbors=3)
        X_train_nearmiss, y_train_nearmiss = near_miss.fit_resample(X_train, y_train)

        print(f"NearMiss后训练集大小: {X_train_nearmiss.shape}, 欺诈比例: {y_train_nearmiss.mean():.2%}")

        # 训练模型
        train_data_nearmiss = lgb.Dataset(X_train_nearmiss, label=y_train_nearmiss)

        model_nearmiss = lgb.train(
            params,
            train_data_nearmiss,
            num_boost_round=500,
            valid_sets=[train_data_nearmiss],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # 预测和评估
        y_pred_proba_nearmiss = model_nearmiss.predict(X_test)
        y_pred_nearmiss = (y_pred_proba_nearmiss > 0.5).astype(int)

        print(f"AUC: {roc_auc_score(y_test, y_pred_proba_nearmiss):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred_nearmiss):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_nearmiss, target_names=['正常', '欺诈']))

        nearmiss_success = True
    except ImportError:
        print("需要安装imbalanced-learn库: pip install imbalanced-learn")
        nearmiss_success = False

    # 方法3.3: Tomek Links欠采样------删除边界上的多数类样本
    print("\n--- 方法3.3: Tomek Links欠采样 ---")

    try:
        from imblearn.under_sampling import TomekLinks

        # 应用Tomek Links
        tomek = TomekLinks()
        X_train_tomek, y_train_tomek = tomek.fit_resample(X_train, y_train)

        print(f"Tomek Links后训练集大小: {X_train_tomek.shape}, 欺诈比例: {y_train_tomek.mean():.2%}")

        # 训练模型
        train_data_tomek = lgb.Dataset(X_train_tomek, label=y_train_tomek)

        model_tomek = lgb.train(
            params,
            train_data_tomek,
            num_boost_round=500,
            valid_sets=[train_data_tomek],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # 预测和评估
        y_pred_proba_tomek = model_tomek.predict(X_test)
        y_pred_tomek = (y_pred_proba_tomek > 0.5).astype(int)

        print(f"AUC: {roc_auc_score(y_test, y_pred_proba_tomek):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred_tomek):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_tomek, target_names=['正常', '欺诈']))

        tomek_success = True
    except ImportError:
        print("需要安装imbalanced-learn库: pip install imbalanced-learn")
        tomek_success = False

    # 方法3.4: 结合过采样和欠采样-------
    print("\n--- 方法3.4: SMOTEENN（结合过采样和欠采样） ---")

    try:
        from imblearn.combine import SMOTEENN

        # 应用SMOTEENN
        smote_enn = SMOTEENN(random_state=42)
        X_train_smoteenn, y_train_smoteenn = smote_enn.fit_resample(X_train, y_train)

        print(f"SMOTEENN后训练集大小: {X_train_smoteenn.shape}, 欺诈比例: {y_train_smoteenn.mean():.2%}")

        # 训练模型
        train_data_smoteenn = lgb.Dataset(X_train_smoteenn, label=y_train_smoteenn)

        model_smoteenn = lgb.train(
            params,
            train_data_smoteenn,
            num_boost_round=500,
            valid_sets=[train_data_smoteenn],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # 预测和评估
        y_pred_proba_smoteenn = model_smoteenn.predict(X_test)
        y_pred_smoteenn = (y_pred_proba_smoteenn > 0.5).astype(int)

        print(f"AUC: {roc_auc_score(y_test, y_pred_proba_smoteenn):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred_smoteenn):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_smoteenn, target_names=['正常', '欺诈']))

        smoteenn_success = True
    except ImportError:
        print("需要安装imbalanced-learn库: pip install imbalanced-learn")
        smoteenn_success = False

    return {
        'random_undersample': (model_downsampled, y_pred_proba_downsampled, y_pred_downsampled),
        'nearmiss': (model_nearmiss, y_pred_proba_nearmiss, y_pred_nearmiss) if nearmiss_success else None,
        'tomek': (model_tomek, y_pred_proba_tomek, y_pred_tomek) if tomek_success else None,
        'smoteenn': (model_smoteenn, y_pred_proba_smoteenn, y_pred_smoteenn) if smoteenn_success else None
    }

if __name__ == "__main__":
    results = method_undersampling()