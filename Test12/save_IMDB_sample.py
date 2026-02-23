##使用 torchtext 0.10.0 加载 IMDB 电影评论数据集，查看原始文本与标签内容，
##并将部分样本保存为 CSV 文件以便人工检查

import torchtext
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer

print("torchtext version:", torchtext.__version__)  # 应显示 0.10.0

# 1. 加载 IMDB 数据集（返回迭代器）
train_iter, test_iter = IMDB(split=('train', 'test'))

# 2. 将迭代器转换为列表，以便多次查看和保存
train_data = list(train_iter)
test_data  = list(test_iter)

print(f"训练集样本数: {len(train_data)}")
print(f"测试集样本数: {len(test_data)}")

# 3. 查看前 5 条训练样本
print("\n=== 前5条训练样本 ===")
for i, (label, text) in enumerate(train_data[:5]):
    print(f"样本 {i+1}: 标签 = {label}")
    print(f"文本预览: {text[:200]}...")  # 只显示前200字符
    print("-" * 60)

# 4. 查看类别分布
from collections import Counter
train_labels = [label for label, _ in train_data]
test_labels  = [label for label, _ in test_data]
print("训练集标签分布:", Counter(train_labels))
print("测试集标签分布:", Counter(test_labels))

# 5. （可选）将部分数据保存为 CSV 文件，便于在 Excel 中查看
try:
    import pandas as pd
    # 选取前100条训练样本
    df_train = pd.DataFrame(train_data, columns=['label', 'text'])
    df_train.to_csv('IMBD_train.csv', index=False, encoding='utf-8')
    print("\n已将前100条训练样本保存至 imdb_train_sample.csv")
except ImportError:
    print("\n未安装 pandas，跳过 CSV 保存。可手动安装: pip install pandas")