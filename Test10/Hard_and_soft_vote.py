"""
投票法（Voting）：分为硬投票和软投票。硬投票是每个模型对样本的预测类别进行投票，票数多的类别作为最终预测类别；
软投票是每个模型对样本的预测类别的概率进行平均，然后取概率最大的类别。
"""
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 创建示例数据
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# 创建特征名称
feature_names = [f'feature_{i}' for i in range(1, X.shape[1] + 1)]
target_name = 'target'

# 创建DataFrame
data = pd.DataFrame(X, columns=feature_names)
data[target_name] = y

# 保存为CSV文件
data.to_csv('dataset.csv', index=False)
print("文件已保存为 'dataset.csv'")
print(f"数据形状: {data.shape}")
print(f"特征数量: {len(feature_names)}")
print(f"目标变量分布:\n{data[target_name].value_counts()}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 初始化基分类器
lr = LogisticRegression(random_state=42, max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)

# 创建硬投票分类器
hard_voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('xgb', xgb)
    ],
    voting='hard'  # 硬投票：选择得票最多的类别
)

# 训练和预测
hard_voting_clf.fit(X_train, y_train)
y_pred_hard = hard_voting_clf.predict(X_test)

print(f"Hard Voting Accuracy: {accuracy_score(y_test, y_pred_hard):.4f}")

#################################
# 创建软投票分类器
soft_voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('xgb', xgb)
    ],
    voting='soft'  # 软投票：基于预测概率的平均值
)

# 需要确保所有分类器都支持predict_proba方法
soft_voting_clf.fit(X_train, y_train)
y_pred_soft = soft_voting_clf.predict(X_test)

print(f"Soft Voting Accuracy: {accuracy_score(y_test, y_pred_soft):.4f}")
