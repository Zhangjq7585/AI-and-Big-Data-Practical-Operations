## 使用网格搜索进行超参数调优

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import time
import numpy as np
import pandas as pd
import Create_data
from Create_data import create_realistic_house_price_dataset

# 读取.xls文件
df = pd.read_excel('Data.xlsx')
# 创建修正后的数据集
X_train, X_test, y_train, y_test, df, feature_names = create_realistic_house_price_dataset()

def grid_search_random_forest(X_train, y_train, X_test, y_test):
    """
    使用网格搜索优化随机森林回归模型
    """
    print("=" * 60)
    print("开始网格搜索超参数调优")
    print("=" * 60)


    # 初始化随机森林回归器
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # 定义超参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],  # 树的数量
        'max_depth': [None, 10, 20, 30],  # 树的最大深度
        'min_samples_split': [2, 5, 10],  # 分裂内部节点所需的最小样本数
        'min_samples_leaf': [1, 2, 4],  # 叶节点所需的最小样本数
        'max_features': ['auto', 'sqrt', 'log2'],  # 寻找最佳分割时考虑的特征数量
        'bootstrap': [True, False]  # 是否使用bootstrap采样
    }

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,  # 5折交叉验证
        scoring='neg_mean_squared_error',  # 使用负均方误差作为评估指标
        n_jobs=-1,  # 使用所有可用的CPU核心
        verbose=1,  # 显示详细进度
        return_train_score=True
    )

    # 记录开始时间
    start_time = time.time()

    # 执行网格搜索
    print("正在执行网格搜索...")
    grid_search.fit(X_train, y_train)

    # 记录结束时间
    end_time = time.time()

    # 输出结果
    print(f"\n网格搜索完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证分数 (负MSE): {grid_search.best_score_:.2f}")

    # 使用最佳模型进行预测
    best_rf = grid_search.best_estimator_
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

    return grid_search, best_rf, y_pred

# 执行网格搜索
grid_result, best_grid_model, y_pred_grid = grid_search_random_forest(X_train, y_train,
                                                                      X_test, y_test)