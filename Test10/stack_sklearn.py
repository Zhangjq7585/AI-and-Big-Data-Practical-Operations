

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
my_estimators = [
    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False))
]

# 创建堆叠分类器
sklearn_stacking = StackingClassifier(
    classifiers=my_estimators,
    meta_classifier=LogisticRegression(random_state=42),
    # cv=5,  # 使用5折交叉验证生成元特征
    # stack_method='predict_proba',  # 使用概率作为元特征
    # passthrough=False  # 是否将原始特征也传递给元分类器
    use_probas=True,    #基础模型输出概率作为元模型特征
    verbose=1
)

# 训练和预测
sklearn_stacking.fit(X_train, y_train)
y_pred_sklearn_stack = sklearn_stacking.predict(X_test)
print(f"Sklearn Stacking Accuracy: {accuracy_score(y_test, y_pred_sklearn_stack):.4f}")
