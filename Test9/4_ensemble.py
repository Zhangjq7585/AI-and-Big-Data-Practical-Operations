
# method4_ensemble.py
"""
方法4：集成方法
使用集成学习技术处理类别不平衡问题
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
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

def method_ensemble():
    """集成方法"""
    print("=" * 60)
    print("方法4: 集成方法")
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

    print(f"训练集大小: {X_train.shape}, 欺诈比例: {y_train.mean():.2%}")

    # 方法4.1: Bagging集成---------自助采样集成
    print("\n--- 方法4.1: Bagging集成 ---")

    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    # 使用决策树作为基分类器
    base_estimator = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    bagging_clf = BaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=50,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=-1,
        random_state=42
    )

    bagging_clf.fit(X_train, y_train)
    y_pred_bagging = bagging_clf.predict(X_test)
    y_pred_proba_bagging = bagging_clf.predict_proba(X_test)[:, 1]

    print(f"AUC: {roc_auc_score(y_test, y_pred_proba_bagging):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_bagging):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_bagging, target_names=['正常', '欺诈']))

    # 方法4.2: 平衡随机森林
    print("\n--- 方法4.2: 平衡随机森林 ---")

    try:
        from imblearn.ensemble import BalancedRandomForestClassifier

        balanced_rf = BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            sampling_strategy='auto',
            replacement=False,
            n_jobs=-1,
            random_state=42
        )

        balanced_rf.fit(X_train, y_train)
        y_pred_balanced_rf = balanced_rf.predict(X_test)
        y_pred_proba_balanced_rf = balanced_rf.predict_proba(X_test)[:, 1]

        print(f"AUC: {roc_auc_score(y_test, y_pred_proba_balanced_rf):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred_balanced_rf):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_balanced_rf, target_names=['正常', '欺诈']))

        balanced_rf_success = True
    except ImportError:
        print("需要安装imbalanced-learn库: pip install imbalanced-learn")
        balanced_rf_success = False

    # 方法4.3: EasyEnsemble--------多个平衡子集的集成
    print("\n--- 方法4.3: EasyEnsemble ---")

    try:
        from imblearn.ensemble import EasyEnsembleClassifier

        easy_ensemble = EasyEnsembleClassifier(
            n_estimators=50,
            estimator=DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            n_jobs=-1,
            random_state=42
        )

        easy_ensemble.fit(X_train, y_train)
        y_pred_easy = easy_ensemble.predict(X_test)
        y_pred_proba_easy = easy_ensemble.predict_proba(X_test)[:, 1]

        print(f"AUC: {roc_auc_score(y_test, y_pred_proba_easy):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred_easy):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_easy, target_names=['正常', '欺诈']))

        easy_ensemble_success = True
    except ImportError:
        print("需要安装imbalanced-learn库: pip install imbalanced-learn")
        easy_ensemble_success = False

    # 方法4.4: RUSBoost-----------结合随机欠采样的Boosting
    print("\n--- 方法4.4: RUSBoost ---")

    try:
        from imblearn.ensemble import RUSBoostClassifier

        rusboost = RUSBoostClassifier(
            n_estimators=50,
            estimator=DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            algorithm='SAMME.R',
            random_state=42
        )
        # 检查并添加可用的参数
        if hasattr(RUSBoostClassifier, '__init__') and 'n_jobs' in RUSBoostClassifier.__init__.__code__.co_varnames:
            rusboost['n_jobs'] = -1

        rusboost.fit(X_train, y_train)
        y_pred_rusboost = rusboost.predict(X_test)
        y_pred_proba_rusboost = rusboost.predict_proba(X_test)[:, 1]

        print(f"AUC: {roc_auc_score(y_test, y_pred_proba_rusboost):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred_rusboost):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_rusboost, target_names=['正常', '欺诈']))

        rusboost_success = True
    except ImportError:
        print("需要安装imbalanced-learn库: pip install imbalanced-learn")
        rusboost_success = False

    # 方法4.5: 基于LightGBM的Stacking集成----多层模型集成
    print("\n--- 方法4.5: 基于LightGBM的Stacking集成 ---")

    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression

    # 创建oof predictions
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 第一层模型预测
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"训练第{fold +1}折...")

        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        oof_predictions[val_idx] = model.predict(X_fold_val)
        test_predictions += model.predict(X_test) / n_splits

    # 第二层模型（元学习器）
    meta_model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )

    # 使用第一层预测作为特征
    X_train_meta = oof_predictions.reshape(-1, 1)
    meta_model.fit(X_train_meta, y_train)

    # 最终预测
    X_test_meta = test_predictions.reshape(-1, 1)
    y_pred_stacking = meta_model.predict(X_test_meta)
    y_pred_proba_stacking = meta_model.predict_proba(X_test_meta)[:, 1]

    print(f"AUC: {roc_auc_score(y_test, y_pred_proba_stacking):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_stacking):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_stacking, target_names=['正常', '欺诈']))

    return {
        'bagging': (bagging_clf, y_pred_proba_bagging, y_pred_bagging),
        'balanced_rf': (balanced_rf, y_pred_proba_balanced_rf, y_pred_balanced_rf) if balanced_rf_success else None,
        'easy_ensemble': (easy_ensemble, y_pred_proba_easy, y_pred_easy) if easy_ensemble_success else None,
        'rusboost': (rusboost, y_pred_proba_rusboost, y_pred_rusboost) if rusboost_success else None,
        'stacking': (meta_model, y_pred_proba_stacking, y_pred_stacking)
    }

if __name__ == "__main__":
    results = method_ensemble()