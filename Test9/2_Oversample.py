# method2_oversampling.py
"""
方法2：过采样方法
通过增加少数类样本来平衡数据集
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

def method_oversampling():
    """过采样方法"""
    print("=" * 60)
    print("方法2: 过采样方法")
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

    # 方法2.1: 随机过采样-----------简单复制少数类样本
    print("\n--- 方法2.1: 随机过采样 ---")

    # 分离多数类和少数类
    X_train_majority = X_train[y_train == 0]
    X_train_minority = X_train[y_train == 1]
    y_train_majority = y_train[y_train == 0]
    y_train_minority = y_train[y_train == 1]

    # 过采样少数类
    from sklearn.utils import resample

    # 将少数类样本数量增加到与多数类相同
    X_train_minority_upsampled, y_train_minority_upsampled = resample(
        X_train_minority,
        y_train_minority,
        replace=True,  # 允许重复采样
        n_samples=len(X_train_majority),  # 增加到多数类数量
        random_state=42
    )

    # 合并过采样后的数据
    X_train_upsampled = pd.concat([X_train_majority, X_train_minority_upsampled])
    y_train_upsampled = pd.concat([y_train_majority, y_train_minority_upsampled])

    # 打乱数据
    shuffle_idx = np.random.permutation(len(X_train_upsampled))
    X_train_upsampled = X_train_upsampled.iloc[shuffle_idx].reset_index(drop=True)
    y_train_upsampled = y_train_upsampled.iloc[shuffle_idx].reset_index(drop=True)

    print(f"过采样后训练集大小: {X_train_upsampled.shape}, 欺诈比例: {y_train_upsampled.mean():.2%}")

    # 训练模型
    train_data = lgb.Dataset(X_train_upsampled, label=y_train_upsampled)

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

    model_upsampled = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # 预测和评估
    y_pred_proba_upsampled = model_upsampled.predict(X_test)
    y_pred_upsampled = (y_pred_proba_upsampled > 0.5).astype(int)

    print(f"AUC: {roc_auc_score(y_test, y_pred_proba_upsampled):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_upsampled):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_upsampled, target_names=['正常', '欺诈']))

    # 方法2.2: SMOTE过采样-----------合成新的少数类样本
    print("\n--- 方法2.2: SMOTE过采样 ---")

    try:
        from imblearn.over_sampling import SMOTE

        # 应用SMOTE
        smote = SMOTE(random_state=42, sampling_strategy=0.5)  # 将少数类增加到50%
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        print(f"SMOTE后训练集大小: {X_train_smote.shape}, 欺诈比例: {y_train_smote.mean():.2%}")

        # 训练模型
        train_data_smote = lgb.Dataset(X_train_smote, label=y_train_smote)

        model_smote = lgb.train(
            params,
            train_data_smote,
            num_boost_round=500,
            valid_sets=[train_data_smote],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # 预测和评估
        y_pred_proba_smote = model_smote.predict(X_test)
        y_pred_smote = (y_pred_proba_smote > 0.5).astype(int)

        print(f"AUC: {roc_auc_score(y_test, y_pred_proba_smote):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred_smote):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_smote, target_names=['正常', '欺诈']))

        smote_success = True
    except ImportError:
        print("需要安装imbalanced-learn库: pip install imbalanced-learn")
        smote_success = False

    # 方法2.3: ADASYN过采样-----------基于样本密度合成新样本
    print("\n--- 方法2.3: ADASYN过采样 ---")

    try:
        from imblearn.over_sampling import ADASYN

        # 应用ADASYN
        adasyn = ADASYN(random_state=42, sampling_strategy=0.5)
        X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

        print(f"ADASYN后训练集大小: {X_train_adasyn.shape}, 欺诈比例: {y_train_adasyn.mean():.2%}")

        # 训练模型
        train_data_adasyn = lgb.Dataset(X_train_adasyn, label=y_train_adasyn)

        model_adasyn = lgb.train(
            params,
            train_data_adasyn,
            num_boost_round=500,
            valid_sets=[train_data_adasyn],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        # 预测和评估
        y_pred_proba_adasyn = model_adasyn.predict(X_test)
        y_pred_adasyn = (y_pred_proba_adasyn > 0.5).astype(int)

        print(f"AUC: {roc_auc_score(y_test, y_pred_proba_adasyn):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred_adasyn):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_adasyn, target_names=['正常', '欺诈']))

        adasyn_success = True
    except ImportError:
        print("需要安装imbalanced-learn库: pip install imbalanced-learn")
        adasyn_success = False

    return {
        'random_oversample': (model_upsampled, y_pred_proba_upsampled, y_pred_upsampled),
        'smote': (model_smote, y_pred_proba_smote, y_pred_smote) if smote_success else None,
        'adasyn': (model_adasyn, y_pred_proba_adasyn, y_pred_adasyn) if adasyn_success else None
    }

if __name__ == "__main__":
    results = method_oversampling()