"""
堆叠法（Stacking）：首先使用多个基模型对训练集进行预测，然后将这些预测结果作为新的特征，
再训练一个元模型（meta-model）来进行最终的预测。堆叠法通常可以融合不同类型的基模型，从而可能获得更好的性能。
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.classifier import StackingClassifier

# 创建示例数据
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 初始化基分类器
lr = LogisticRegression(random_state=42, max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)

# 元分类器（最终分类器）
meta_classifier = LogisticRegression(random_state=42, max_iter=1000)

# 创建堆叠分类器
stacking_clf = StackingClassifier(
    classifiers=[lr, rf, xgb],
    meta_classifier=meta_classifier,
    use_probas=True,      # 使用概率作为元特征
    average_probas=False, # 不平均概率
    verbose=1             # 显示训练过程
)

# 训练堆叠分类器
stacking_clf.fit(X_train, y_train)

# 预测
y_pred_stack = stacking_clf.predict(X_test)
print(f"Stacking Accuracy: {accuracy_score(y_test,y_pred_stack):.4f}")

# 查看基分类器的预测结果（元特征）
meta_features = stacking_clf.predict_meta_features(X_test)
print(f"Meta-features shape: {meta_features.shape}")

