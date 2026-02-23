###########
'''
 问题答案：
    1.职业字段：属于无顺序的名义变量，适合独热编码(One-Hot Encoding),避免给不同职业赋予隐含的大小关系；
    2.”学历“字段：属于有顺序的orignal变量（大专<本科<硕士<博士），适合使用标签编码(Label Encoding),
    保留学历的等级关系。
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# 示例数据
data = pd.DataFrame({
    '职业': ['教师', '医生', '工程师', '程序员', '医生', '教师', '工程师'],
    '学历': ['本科', '硕士', '博士', '本科', '大专', '硕士', '博士']
})

print("原始数据：")
print(data)
print()

# 方法1：独热编码（推荐）
# 对职业进行独热编码（名义变量）
occupation_dummies = pd.get_dummies(data['职业'], prefix='职业')
# 对学历进行序数编码（有序变量）
education_order = {'大专': 1, '本科': 2, '硕士': 3, '博士': 4}
data['学历_编码'] = data['学历'].map(education_order)

# 合并结果
data_encoded = pd.concat([data, occupation_dummies], axis=1)
print("方法1：独热编码+序数编码结果：")
print(data_encoded)
print()

# 方法2：全部使用独热编码
data_encoded_full = pd.get_dummies(data, columns=['职业', '学历'])
print("方法2：全部独热编码结果：")
print(data_encoded_full)
print()

# 方法3：使用标签编码（适合树模型）
label_encoder = LabelEncoder()
data['职业_标签编码'] = label_encoder.fit_transform(data['职业'])
print("方法3：标签编码结果：")
print(data[['职业', '职业_标签编码']].drop_duplicates())



