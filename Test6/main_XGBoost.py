## XGBoost模型

# 3. XGBoost模型（梯度提升）
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 设置全局字体为支持中文的字体，例如使用SimHei（黑体）.防止中文显示乱码
plt.rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei', 'Arial Unicode MS' 等
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 导入.csv数据
df = pd.read_csv('DATA.csv')

# 数据准备
X = df.drop(['customer_id', 'churn'], axis=1)
y = df['churn']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# 3. 定义预处理管道 - preprocessor就是在这里定义的
# 分离数值型和分类型特征
numeric_features = ['tenure', 'monthly_charges', 'total_charges', 'age', 'dependents']
categorical_features = ['contract_type', 'payment_method', 'internet_service']

# 预处理数据（转换为XGBoost可处理的格式）
preprocessor_only = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# 转换数据
X_processed = preprocessor_only.fit_transform(X)

# 重新划分训练集和测试集
X_train_proc, X_test_proc, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# 创建XGBoost模型
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # 处理类别不平衡
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 使用交叉验证训练
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_train_proc, y_train,
                           cv=cv, scoring='roc_auc')
print(f"\nXGBoost交叉验证AUC得分: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# 训练最终模型
xgb_model.fit(X_train_proc, y_train)

# 预测
y_pred_xgb = xgb_model.predict(X_test_proc)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_proc)[:, 1]

# 模型评估
print("\n" + "=" * 60)
print("XGBoost模型评估")
print("=" * 60)

print("\n1. 分类报告:")
print(classification_report(y_test, y_pred_xgb, target_names=['未流失', '流失']))

print("\n2. AUC-ROC得分:", roc_auc_score(y_test, y_pred_proba_xgb))

# 混淆矩阵
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Reds',
            xticklabels=['未流失', '流失'],
            yticklabels=['未流失', '流失'])
plt.title('XGBoost - 混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

## 导入 main_randomforest.py 的变量
import main_randomforest

# 比较两个模型的ROC曲线
fpr_rf, tpr_rf, _ = roc_curve(y_test, main_randomforest.y_pred_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)

plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC = {roc_auc_score(y_test, main_randomforest.y_pred_proba_rf):.3f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, y_pred_proba_xgb):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
plt.xlabel('假正率 (FPR)')
plt.ylabel('真正率 (TPR)')
plt.title('模型ROC曲线对比')
plt.legend()
plt.grid(True)
plt.show()

# 性能对比
results_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [
        (main_randomforest.y_pred_rf == y_test).mean(),
        (y_pred_xgb == y_test).mean()
    ],
    'Precision (流失类)': [
        confusion_matrix(y_test, main_randomforest.y_pred_rf)[1, 1] / confusion_matrix(y_test, main_randomforest.y_pred_rf)[:, 1].sum(),
        confusion_matrix(y_test, y_pred_xgb)[1, 1] / confusion_matrix(y_test, y_pred_xgb)[:, 1].sum()
    ],
    'Recall (流失类)': [
        confusion_matrix(y_test, main_randomforest.y_pred_rf)[1, 1] / confusion_matrix(y_test, main_randomforest.y_pred_rf)[1, :].sum(),
        confusion_matrix(y_test, y_pred_xgb)[1, 1] / confusion_matrix(y_test, y_pred_xgb)[1, :].sum()
    ],
    'F1-Score (流失类)': [
        2 * confusion_matrix(y_test, main_randomforest.y_pred_rf)[1, 1] /
        (2 * confusion_matrix(y_test, main_randomforest.y_pred_rf)[1, 1] +
         confusion_matrix(y_test, main_randomforest.y_pred_rf)[0, 1] +
         confusion_matrix(y_test, main_randomforest.y_pred_rf)[1, 0]),
        2 * confusion_matrix(y_test, y_pred_xgb)[1, 1] /
        (2 * confusion_matrix(y_test, y_pred_xgb)[1, 1] +
         confusion_matrix(y_test, y_pred_xgb)[0, 1] +
         confusion_matrix(y_test, y_pred_xgb)[1, 0])
    ],
    'AUC-ROC': [
        roc_auc_score(y_test, main_randomforest.y_pred_proba_rf),
        roc_auc_score(y_test, y_pred_proba_xgb)
    ]
})

print("\n" + "=" * 60)
print("模型性能对比")
print("=" * 60)
print(results_comparison.round(3))


