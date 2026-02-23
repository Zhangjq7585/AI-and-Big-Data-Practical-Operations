## 使用随机森林模型
'''
优点：
    ✅ 处理非线性：决策树天然捕捉非线性关系
    ✅ 特征重要性：提供特征排序，帮助业务理解
    ✅ 稳健性强：对异常值、缺失值不敏感
    ✅ 抗过拟合：集成+随机采样减少过拟合风险
    ✅ 无需复杂预处理：可处理数值/类别混合特征
缺点：
    ❌ 可解释性差：相比逻辑回归难解释
    ❌ 内存占用大：存储多棵树，内存消耗大
    ❌ 预测速度慢：需要遍历多棵树做预测
    ❌ 可能过拟合：树深度过大或数据太少时仍可能过拟合
    ❌ 外推能力差：对超出训练范围的数据预测不准
适用场景： 中等规模数据、需要特征重要性、非线性关系复杂
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, average_precision_score

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

# 3. 定义预处理管道 - preprocessor就是在这里定义的
# 分离数值型和分类型特征
numeric_features = ['tenure', 'monthly_charges', 'total_charges', 'age', 'dependents']
categorical_features = ['contract_type', 'payment_method', 'internet_service']

# 数值型特征的处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 处理缺失值
    ('scaler', StandardScaler())  # 标准化
])
# 分类型特征的处理管道
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 处理缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 独热编码
])

# 这里定义preprocessor - 使用ColumnTransformer组合不同类型特征的预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 创建随机森林模型管道
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ))
])

# 超参数网格（为了演示，使用简化的网格）
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# 网格搜索（为了速度，可以设置cv=3）
rf_grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

# 训练模型（使用网格搜索）
rf_grid_search.fit(X_train, y_train)

# 最佳模型
best_rf = rf_grid_search.best_estimator_

# 预测
y_pred_rf = best_rf.predict(X_test)
y_pred_proba_rf = best_rf.predict_proba(X_test)[:, 1]

# 评估模型
print("=" * 60)
print("随机森林模型评估")
print("=" * 60)
print(f"最佳参数: {rf_grid_search.best_params_}")

# 准确率
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"准确率: {accuracy_rf:.4f}")

# AUC分数
auc_score_rf = roc_auc_score(y_test, y_pred_proba_rf)
print(f"AUC分数: {auc_score_rf:.4f}")

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred_rf, target_names=['未流失', '流失']))

# 混淆矩阵
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['未流失', '流失'],
            yticklabels=['未流失', '流失'])
plt.title('随机森林 - 混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

## 导入 main_logic_regression.py 的变量
import main_logic_regression
# fpr=[0.        , 0.00606061, 0.00606061, 0.03030303, 0.03030303, 0.04848485,
#      0.04848485, 0.06060606, 0.06060606, 0.06666667, 0.06666667, 0.08484848,
#      0.08484848, 0.09090909, 0.09090909, 0.10909091, 0.10909091, 0.11515152,
#      0.11515152, 0.12727273, 0.12727273, 0.13939394, 0.13939394, 0.15151515,
#      0.15151515, 0.18181818, 0.18181818, 0.24242424, 0.24242424, 0.27272727,
#      0.27272727, 0.27878788,0.27878788, 0.33939394, 0.33939394, 0.37575758,
#      0.37575758, 0.43636364, 0.43636364, 1.        ]
# tpr=[0.        , 0.        , 0.05714286, 0.05714286,0.2,        0.2,
#      0.28571429, 0.28571429, 0.31428571, 0.31428571, 0.45714286, 0.45714286,
#      0.51428571, 0.51428571, 0.54285714, 0.54285714, 0.57142857, 0.57142857,
#      0.62857143, 0.62857143, 0.65714286, 0.65714286, 0.68571429, 0.68571429,
#      0.74285714, 0.74285714, 0.77142857, 0.77142857, 0.8,        0.8,
#      0.82857143, 0.82857143, 0.91428571, 0.91428571, 0.94285714, 0.94285714,
#      0.97142857, 0.97142857, 1.        , 1.        ]
auc_score=0.8697835497835498
# ROC曲线
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_proba_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'随机森林 (AUC = {auc_score_rf:.3f})')
plt.plot(main_logic_regression.fpr, main_logic_regression.tpr, color='darkorange', lw=2,
         linestyle='--', label=f'逻辑回归 (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':', label='随机猜测')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('模型比较 - ROC曲线')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# 精确率-召回率曲线（特别适用于不平衡数据）
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_rf)
average_precision = average_precision_score(y_test, y_pred_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2,
         label=f'随机森林 (AP = {average_precision:.3f})')
plt.xlabel('召回率 (Recall)')
plt.ylabel('精确率 (Precision)')
plt.title('随机森林 - 精确率-召回率曲线')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.show()

# 特征重要性（随机森林）
rf_classifier = best_rf.named_steps['classifier']
# 获取特征名称
preprocessor.fit(X_train)
feature_names = (numeric_features +
                 list(rf_pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features)))

feature_importance_rf = pd.DataFrame({
    '特征': feature_names,
    '重要性': rf_classifier.feature_importances_
}).sort_values('重要性', ascending=False)

print("\nTop 10特征重要性（随机森林）:")
print(feature_importance_rf.head(10))

# accuracy=0.76
# y_pred=[0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
#         1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0,
#         0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1,
#         0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
#         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1,
#         0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
# 模型比较总结
print("\n" + "=" * 60)
print("模型性能总结")
print("=" * 60)

summary = pd.DataFrame({
    '模型': ['逻辑回归', '随机森林'],
    '准确率': [main_logic_regression.accuracy, accuracy_rf],
    'AUC分数': [auc_score, auc_score_rf],
    '召回率(流失类)': [
        classification_report(y_test, main_logic_regression.y_pred, output_dict=True, target_names=['未流失', '流失'])['流失']['recall'],
        classification_report(y_test, y_pred_rf, output_dict=True, target_names=['未流失', '流失'])['流失']['recall']
    ],
    '精确率(流失类)': [
        classification_report(y_test, main_logic_regression.y_pred, output_dict=True, target_names=['未流失', '流失'])['流失']['precision'],
        classification_report(y_test, y_pred_rf, output_dict=True, target_names=['未流失', '流失'])['流失']['precision']
    ]
})
print(summary)