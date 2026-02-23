# 完整的缺失值和异常值处理流程
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def preprocess_ecommerce_data(file_path):
    """
    完整的电商数据预处理函数
    包含：年龄缺失值填充 + 消费金额异常值处理
    """
    # 1. 读取数据
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df['消费时间'] = pd.to_datetime(df['消费时间'])

    # 2. 处理年龄缺失值（使用回归模型）
    print("步骤1：处理年龄缺失值...")
    if df['年龄'].isna().any():
        # 准备特征
        df['性别编码'] = (df['性别'] == '男').astype(int)
        df['消费月份'] = df['消费时间'].dt.month
        df['消费星期'] = df['消费时间'].dt.weekday

        # 分离数据
        known_age = df[df['年龄'].notna()]
        unknown_age = df[df['年龄'].isna()]

        if len(unknown_age) > 0:
            features = ['消费金额', '性别编码', '消费月份', '消费星期']
            X_train = known_age[features]
            y_train = known_age['年龄']
            X_predict = unknown_age[features]

            # 训练模型
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # 预测并填充
            predicted_ages = model.predict(X_predict)
            predicted_ages = np.round(predicted_ages).astype(int)
            predicted_ages = np.clip(predicted_ages, 18, 65)

            df.loc[unknown_age.index, '年龄'] = predicted_ages

        # 清理临时列
        df = df.drop(['性别编码', '消费月份', '消费星期'], axis=1)

    # 3. 处理消费金额异常值
    print("步骤2：处理消费金额异常值...")

    # 定义异常阈值（基于业务或统计）
    threshold = 5000  # 业务阈值

    # 检测异常值
    outliers = df['消费金额'] > threshold

    if outliers.any():
        # 计算合理上限（99分位数）
        upper_limit = df.loc[~outliers, '消费金额'].quantile(0.99)

        # 截断异常值
        df.loc[outliers, '消费金额'] = upper_limit

        print(f"  检测到 {outliers.sum()} 个异常值")
        print(f"  已将大于 {threshold} 的值截断为 {upper_limit:.2f}")

    # 4. 可选：对数变换
    df['消费金额_log'] = np.log1p(df['消费金额'])

    print("✅ 预处理完成！")
    print(f"  总记录数：{len(df)}")
    print(f"  年龄缺失值：{df['年龄'].isna().sum()}（已全部处理）")
    print(f"  消费金额范围：{df['消费金额'].min():.2f} - {df['消费金额'].max():.2f}")

    return df

# 使用示例
processed_df = preprocess_ecommerce_data('Dataset/ecommerce_consume.csv')
processed_df.to_csv('Dataset/ecommerce_consume_after.csv', index=False, encoding='utf-8-sig')