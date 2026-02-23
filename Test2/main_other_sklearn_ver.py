## 兼容性更好的版本（适用于不同版本的scikit-learn）

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def encode_features_safe(data):
    """兼容性更好的编码函数"""

    # 检查scikit-learn版本
    import sklearn
    version = sklearn.__version__
    print(f"scikit-learn版本: {version}")

    # 备份原始数据
    df = data.copy()

    # 方法A：使用sparse参数（兼容旧版本）
    try:
        # 新版本
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    except TypeError:
        try:
            # 尝试旧版本参数
            ohe = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        except TypeError:
            # 更旧的版本
            ohe = OneHotEncoder(drop='first', handle_unknown='ignore')

    # 创建列转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('occupation', ohe, ['职业']),
            ('education', OrdinalEncoder(categories=[['大专', '本科', '硕士', '博士']]), ['学历'])
        ],
        remainder='passthrough'
    )

    # 转换数据
    transformed = preprocessor.fit_transform(df)

    # 获取特征名（兼容性处理）
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # 旧版本scikit-learn
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'occupation':
                # 获取独热编码的列名
                categories = trans.categories_[0]
                if trans.drop is not None:
                    categories = categories[1:]  # drop='first'
                feature_names.extend([f'{name}_{cat}' for cat in categories])
            elif name == 'education':
                feature_names.append('education_encoded')
            else:
                feature_names.extend(cols)

    # 创建结果DataFrame
    result_df = pd.DataFrame(transformed, columns=feature_names)

    return result_df, preprocessor

# 使用示例
data = pd.DataFrame({
    '职业': ['教师', '医生', '工程师', '程序员', '医生', '教师', '工程师'],
    '学历': ['本科', '硕士', '博士', '本科', '大专', '硕士', '博士'],
    '年龄': [28, 35, 32, 26, 40, 29, 33]
})

encoded_df, encoder = encode_features_safe(data)
print("兼容性版本编码结果：")
print(encoded_df)
print("\n编码器信息：", encoder)