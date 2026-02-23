

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

print("\n" + "="*50)
print("分别编码后合并")

# 示例数据
data = pd.DataFrame({
    '职业': ['教师', '医生', '工程师', '程序员', '医生', '教师', '工程师'],
    '学历': ['本科', '硕士', '博士', '本科', '大专', '硕士', '博士'],
    '其他特征': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8]  # 数值特征示例
})

# 1. 对职业进行独热编码
try:
    ohe = OneHotEncoder(drop='first', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(drop='first', sparse=False)

occupation_encoded = ohe.fit_transform(data[['职业']])
occupation_cols = [f'职业_{cat}' for cat in ohe.categories_[0][1:]]
occupation_df = pd.DataFrame(occupation_encoded, columns=occupation_cols)

# 2. 对学历进行序数编码
education_categories = [['大专', '本科', '硕士', '博士']]
ordinal_encoder = OrdinalEncoder(categories=education_categories)
education_encoded = ordinal_encoder.fit_transform(data[['学历']])
education_df = pd.DataFrame(education_encoded, columns=['学历_编码'])

# 3. 获取其他特征
other_features = data.drop(['职业', '学历'], axis=1)

# 4. 合并所有特征
final_df = pd.concat([education_df, occupation_df, other_features], axis=1)

print("\n分别编码合并结果：")
print(final_df)
print("\n列信息：")
for col in final_df.columns:
    print(f"  {col}")


