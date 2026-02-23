## 逻辑回归模型
'''
优点：
    ✅ 可解释性强：特征系数直接反映影响方向与大小
    ✅ 计算效率高：训练和预测速度快
    ✅ 输出概率：天然输出流失概率，易于设定阈值
    ✅ 稳定性好：对噪声特征不敏感，不易过拟合
    ✅ 可添加正则化：L1/L2正则防止过拟合，L1还能做特征选择
缺点：
    ❌ 线性假设强：难以捕捉非线性关系和特征交互
    ❌ 特征工程要求高：需要手动创建交互项、多项式特征
    ❌ 对不平衡数据敏感：需搭配class_weight或采样技术
    ❌ 异常值敏感：需要预处理异常值

适用场景： 基线模型、可解释性要求高、特征基本线性相关
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体为支持中文的字体，例如使用SimHei（黑体）.防止中文显示乱码
plt.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei', 'Arial Unicode MS' 等
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 导入.csv数据
df = pd.read_csv('DATA.csv')

# 数据准备
X = df.drop(['customer_id', 'churn'], axis=1)
y = df['churn']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 预处理管道
numeric_features = ['tenure', 'monthly_charges', 'total_charges', 'age', 'dependents']
categorical_features = ['contract_type', 'payment_method', 'internet_service']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
# ])    ## 报错！！！

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 创建逻辑回归模型管道
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'  # 处理类别不平衡
    ))
])

# 训练模型
logreg_pipeline.fit(X_train, y_train)

# 预测
y_pred = logreg_pipeline.predict(X_test)
y_pred_proba = logreg_pipeline.predict_proba(X_test)[:, 1]

# 评估模型
print("=" * 60)
print("逻辑回归模型评估")
print("=" * 60)

# 准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# AUC分数
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC分数: {auc_score:.4f}")

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['未流失', '流失']))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['未流失', '流失'],
            yticklabels=['未流失', '流失'])
plt.title('逻辑回归 - 混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('逻辑回归 - ROC曲线')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# 特征重要性（逻辑回归的系数）
# 获取特征名称
preprocessor.fit(X_train)
feature_names = (numeric_features +
                 list(logreg_pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features)))

# 获取系数
coefficients = logreg_pipeline.named_steps['classifier'].coef_[0]

# 创建特征重要性DataFrame
feature_importance = pd.DataFrame({
    '特征': feature_names,
    '系数': coefficients
}).sort_values('系数', ascending=False)

print("\nTop 10特征重要性（逻辑回归系数）:")
print(feature_importance.head(10))