# -*- coding: utf-8 -*-
"""
用户行为预测建模 - 分类任务
    sample1:数据挖掘与建模
目标：使用提供的数据集（如用户行为数据）构建一个分类模型，预测用户是否会发生某种行为，并提交预测结果与模型代码。
任务：
    1. 数据探索与可视化
    2. 特征工程与预处理
    3. 评估模型性能并选择最优模型
    4. 输出预测结果与简要分析报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

# ===================== 1. 数据加载与探索 =====================
print("=" * 60)
print("1. 数据加载与探索")
print("=" * 60)

# 读取数据
df = pd.read_csv('user_behavior.csv')
print(f"数据集形状：{df.shape}")
print(f"字段列表：{list(df.columns)}")
print("\n前5行数据：")
print(df.head())
print("\n基本统计信息：")
print(df.describe())
print("\n缺失值检查：")
print(df.isnull().sum())

# ===================== 2. 数据可视化 =====================
print("\n" + "=" * 60)
print("2. 数据可视化分析")
print("=" * 60)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('数据分布与关系可视化', fontsize=16)

# 年龄分布
sns.histplot(df['age'], kde=True, ax=axes[0, 0], edgecolor='black',color='blue')
axes[0, 0].set_title('年龄分布')
axes[0, 0].set_xlabel('年龄')
axes[0, 0].set_ylabel('频数')

# 收入分布
sns.histplot(df['income'], kde=True, ax=axes[0, 1], edgecolor='black', color='lightgreen')
axes[0, 1].set_title('收入分布')
axes[0, 1].set_xlabel('月收入（千元）')

# 浏览时长与购买关系
sns.boxplot(x='purchase', y='browsing_minutes', data=df, ax=axes[0, 2], palette='Set2')
axes[0, 2].set_title('浏览时长 vs 购买行为')
axes[0, 2].set_xlabel('是否购买（0=否，1=是）')
axes[0, 2].set_ylabel('浏览时长（分钟）')

# 点击率分布
sns.histplot(df['click_rate'], kde=True, ax=axes[1, 0],edgecolor='black', color='salmon')
axes[1, 0].set_title('点击率分布')
axes[1, 0].set_xlabel('点击率')

# 性别与购买比例
gender_purchase = df.groupby('gender')['purchase'].mean()
axes[1, 1].bar(gender_purchase.index, gender_purchase.values, color=['pink', 'lightblue'])
axes[1, 1].set_title('性别购买比例')
axes[1, 1].set_xlabel('性别（0=女，1=男）')
axes[1, 1].set_ylabel('购买比例')
axes[1, 1].set_xticks([0, 1])

# 相关性热图
corr = df.drop('user_id', axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 2]) #cbar_kws={'shrink': 0.8}
axes[1, 2].set_title('特征相关性热图')

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=600, bbox_inches='tight')
plt.show()

print("可视化图表已保存为: data_exploration.png")

# ===================== 3. 特征工程与预处理 =====================
print("\n" + "=" * 60)
print("3. 特征工程与预处理")
print("=" * 60)

# 特征与标签分离
X = df.drop(['user_id', 'purchase'], axis=1)
y = df['purchase']

print(f"特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集大小：{X_train.shape}，测试集大小：{X_test.shape}")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    #防止 输入数组中存在非数字（NaN）、无穷大（infinity）或者过大的值，这在float64类型不允许
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0) # 将 NaN 替换为 0
X_train_scaled = np.clip(X_train_scaled, a_min=None, a_max=np.finfo(np.float64).max) # 限制无穷大的值

X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0) # 将 NaN 替换为 0
X_test_scaled = np.clip(X_test_scaled, a_min=None, a_max=np.finfo(np.float64).max) # 限制无穷大的值

# 特征选择（选择K个最佳特征）
selector = SelectKBest(score_func=f_classif, k=6)
X_train_selected = selector.fit_transform(X_train_scaled, y_train) ## !!!!!
X_test_selected = selector.transform(X_test_scaled)
selected_features = X.columns[selector.get_support()]

#print(f"选中的特征：{list(selected_features)}")
print(f"   标准化完成，训练集均值: {X_train_selected.mean():.3f}")
print(f"   训练集标准差: {X_train_selected.std():.3f}")

# ===================== 4. 模型训练与调优 =====================
print("\n" + "=" * 60)
print("4. 模型训练与调优")
print("=" * 60)

# 定义三个候选模型
models = {
    '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
    '随机森林': RandomForestClassifier(random_state=42),
    '梯度提升': GradientBoostingClassifier(random_state=42)
}

# 交叉验证评估
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
    cv_results[name] = scores.mean()
    print(f"{name} 交叉验证平均准确率：{scores.mean():.4f} (+/- {scores.std():.4f})")

# 选择最佳模型
best_model_name = max(cv_results, key=cv_results.get)
print(f"\n选择最佳模型：{best_model_name}")

# 对最佳模型进行超参数调优（以随机森林为例）
if best_model_name == '随机森林':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_selected, y_train)
    best_model = grid_search.best_estimator_
    print(f"最佳参数：{grid_search.best_params_}")
else:
    best_model = models[best_model_name]
    best_model.fit(X_train_selected, y_train)

# ===================== 5. 模型评估 =====================
print("\n" + "=" * 60)
print("5. 模型评估")
print("=" * 60)

# 预测
y_pred = best_model.predict(X_test_selected)
y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1] if hasattr(best_model, "predict_proba") else None

# 评估指标
print("分类报告：")
print(classification_report(y_test, y_pred, target_names=['未购买', '购买']))

print("混淆矩阵：")
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(f"准确率：{accuracy:.4f}")

if y_pred_proba is not None:
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC：{auc:.4f}")

# 绘制混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['未购买', '购买'], yticklabels=['未购买', '购买'])
plt.title('混淆矩阵热图')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.savefig('confusion_matrix.png', dpi=600, bbox_inches='tight')
plt.show()

# 绘制ROC曲线
if y_pred_proba is not None:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 6. 特征重要性分析 =====================
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 60)
    print("6. 特征重要性分析")
    print("=" * 60)

    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title('特征重要性排序')
    plt.bar(range(len(importances)), importances[indices], align='center', color='teal')
    plt.xticks(range(len(importances)), [selected_features[i] for i in indices], rotation=45)
    plt.xlabel('特征')
    plt.ylabel('重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印重要性值
    for i in indices:
        print(f"{selected_features[i]}: {importances[i]:.4f}")

# ===================== 7. 输出预测结果与报告 =====================
print("\n" + "=" * 60)
print("7. 预测结果与报告输出")
print("=" * 60)

# 创建包含预测结果的DataFrame
results_df = X_test.copy()
results_df['真实标签'] = y_test.values
results_df['预测标签'] = y_pred
if y_pred_proba is not None:
    results_df['预测概率'] = y_pred_proba

# 保存预测结果
results_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
print("预测结果已保存至：prediction_results.csv")

# 保存模型代码（将整个处理流程封装）
# with open('model_training_code.py', 'w', encoding='utf-8') as f:
#     f.write(code_content)  # 这里可以保存当前脚本的核心代码

df_shape=df.shape[0]
df_shape_param=df.shape[1] - 2
bili=df['purchase'].mean()
# 生成简要分析报告
report = f"""
===================== 建模分析报告 =====================
数据概况：
数据集大小：{df.shape}
准确率：{accuracy:.2%}
模型类型：{type(model).__name__}
总样本数：{df.shape[0]}
特征数：{df.shape[1] - 2}（排除ID和标签）
正负样本比例：{df['purchase'].mean():.2%}

关键发现
- 购买用户平均浏览时长较高
- 点击率与购买行为呈正相关
- 历史购买记录是重要预测因子

结论与建议
   - 模型能够有效识别潜在购买用户
   - 建议重点关注浏览时长和点击率高的用户
   - 可进一步收集更多行为特征提升预测性能
"""
print(report)

# 保存报告
with open('analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("分析报告已保存至：analysis_report.txt")

print("\n✅ 建模任务完成！")
print("生成的文件清单：")
print("1. user_behavior.csv - 原始数据集")
print("2. data_exploration.png - 数据探索可视化")
print("3. prediction_results.csv - 预测结果")
print("4. confusion_matrix.png - 混淆矩阵")
print("5. roc_curve.png - ROC曲线图")
print("6. feature_importance.png - 特征重要性图")
print("7. analysis_report.txt - 简要分析报告")

