# test5:计算滑动窗口，构建时间特征。
# 	利用Pandas的滚动窗口(rolling)函数，按"日期"排序后，分别计算7天窗口的均值和3天窗口的最大值，
# 	构建时间特征。

import pandas as pd
import numpy as np

# 创建示例时间序列数据集
np.random.seed(42)
date_range = pd.date_range(start='2024-01-01', end='2024-02-29', freq='D')  #构建时间序列
sales_data = np.random.randint(1000, 5000, size=len(date_range))            #构建销售金额

df = pd.DataFrame({
    '日期': date_range,
    '日销售额': sales_data
})

print("原始数据集示例:")
print(df.head(10))
print(f"\n数据集形状: {df.shape}")

df.to_csv('DATA.csv', index=False)  #保存原始数据集
# 最简洁的实现方式
def create_sliding_window_features(df, date_col='日期', value_col='日销售额'):
    """
    创建滑动窗口特征的简洁版本
    """
    # 复制数据，避免修改原数据
    result = df.copy()

    # 按日期排序
    result = result.sort_values(date_col).reset_index(drop=True)

    # 计算滑动窗口特征
    # 过去7天平均销售额（不包括当天）
    result['过去7天平均销售额'] = (
        result[value_col]
        .rolling(window=7, min_periods=1)
        .mean()
        .shift(1)
    )

    # 过去3天销售额最大值（不包括当天）
    result['过去3天销售额最大值'] = (
        result[value_col]
        .rolling(window=3, min_periods=1)
        .max()
        .shift(1)
    )

    return result


# 使用示例
final_df = create_sliding_window_features(df)
print("\n最终数据集（含时间特征）:")
print(final_df[['日期', '日销售额', '过去7天平均销售额', '过去3天销售额最大值']].head(12))

final_df.to_csv("Final_DATA.csv", index=False)  #保存最终销售额的处理结果
