## 使用随机搜索进行超参数调优

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import numpy as np
import pandas as pd
from Create_data import create_realistic_house_price_dataset
import grid_search

# 读取.xls文件
df = pd.read_excel('Data.xlsx')
# 创建修正后的数据集
X_train, X_test, y_train, y_test, df, feature_names = create_realistic_house_price_dataset()

def random_search_random_forest(X_train, y_train, X_test, y_test, n_iter=50):
    """
    使用随机搜索优化随机森林回归模型
    """
    print("\n" + "=" * 60)
    print("开始随机搜索超参数调优")
    print("=" * 60)

    # 初始化随机森林回归器
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # 定义参数分布（使用分布而不是网格）
    param_dist = {
        'n_estimators': randint(50, 500),  # 树的数量：50到500之间的随机整数
        'max_depth': [None] + list(randint(5, 50).rvs(10)),  # 最大深度
        'min_samples_split': randint(2, 20),  # 分裂内部节点所需的最小样本数
        'min_samples_leaf': randint(1, 10),  # 叶节点所需的最小样本数
        'max_features': ['auto', 'sqrt', 'log2', None],  # 考虑的特征数量
        'bootstrap': [True, False],  # 是否使用bootstrap采样
        'max_samples': uniform(0.5, 0.5)  # 从样本中抽取的比例
    }

    # 创建RandomizedSearchCV对象
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,  # 随机采样的参数组合数量
        cv=5,  # 5折交叉验证
        scoring='neg_mean_squared_error',  # 使用负均方误差作为评估指标
        n_jobs=-1,  # 使用所有可用的CPU核心
        verbose=1,  # 显示详细进度
        random_state=42,  # 确保可重复性
        return_train_score=True
    )

    # 记录开始时间
    start_time = time.time()

    # 执行随机搜索
    print(f"正在执行随机搜索（{n_iter}次迭代）...")
    random_search.fit(X_train, y_train)

    # 记录结束时间
    end_time = time.time()

    # 输出结果
    print(f"\n随机搜索完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"最佳参数组合: {random_search.best_params_}")
    print(f"最佳交叉验证分数 (负MSE): {random_search.best_score_:.2f}")

    # 使用最佳模型进行预测
    best_rf = random_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    # 计算测试集上的性能指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n测试集性能评估:")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"R²分数: {r2:.4f}")

    feature_names = [
        '房屋面积(平方米)',
        '房间数',
        '浴室数',
        '楼层数',
        '建造年份',
        '地理位置评分',
        '地铁距离(公里)',
        '学校评分'
    ]
    # 输出特征重要性
    feature_importance = pd.DataFrame({
        '特征': feature_names,
        '重要性': best_rf.feature_importances_
    }).sort_values('重要性', ascending=False)

    print(f"\n特征重要性排序:")
    print(feature_importance)

    # 比较网格搜索和随机搜索的结果
    print(f"\n{'=' * 60}")
    print("网格搜索 vs 随机搜索 对比")
    print(f"{'=' * 60}")
    print(f"网格搜索最佳RMSE: {np.sqrt(-grid_search.grid_result.best_score_):.2f}")
    print(f"随机搜索最佳RMSE: {np.sqrt(-random_search.best_score_):.2f}")

    return random_search, best_rf, y_pred


# 执行随机搜索
random_result, best_random_model, y_pred_random = random_search_random_forest(
    X_train, y_train, X_test, y_test, n_iter=50
)

# 可视化比较两种方法的结果
import matplotlib.pyplot as plt


def visualize_results(y_test, y_pred_grid, y_pred_random):
    """
    可视化两种搜索方法的预测结果对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 预测值与实际值散点图
    axes[0].scatter(y_test, y_pred_grid, alpha=0.5, label='网格搜索')
    axes[0].scatter(y_test, y_pred_random, alpha=0.5, label='随机搜索')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('实际房价')
    axes[0].set_ylabel('预测房价')
    axes[0].set_title('预测值与实际值对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 残差图
    residuals_grid = y_test - y_pred_grid
    residuals_random = y_test - y_pred_random
    axes[1].scatter(y_pred_grid, residuals_grid, alpha=0.5, label='网格搜索')
    axes[1].scatter(y_pred_random, residuals_random, alpha=0.5, label='随机搜索')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('预测房价')
    axes[1].set_ylabel('残差')
    axes[1].set_title('残差图')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 误差分布
    axes[2].hist(residuals_grid, bins=30, alpha=0.5, label='网格搜索')
    axes[2].hist(residuals_random, bins=30, alpha=0.5, label='随机搜索')
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[2].set_xlabel('残差')
    axes[2].set_ylabel('频率')
    axes[2].set_title('残差分布')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('两种方法的预测结果对比.png', dpi=600)
    plt.show()


# 可视化结果对比
visualize_results(y_test, grid_search.y_pred_grid, y_pred_random)