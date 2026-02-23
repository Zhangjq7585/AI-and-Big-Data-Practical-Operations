# 将 IMDB 数据集的 .npz 和 .json 文件转换为可读的文本文件，方便查看评论内容及其情感标签。
# 代码会输出训练集和测试集的前 20 个样本到 imdb_samples.txt

import json
import numpy as np

# 路径配置
data_path = r"E:\Download\imdb"
npz_file = f"{data_path}/imdb.npz"
word_index_file = f"{data_path}/imdb_word_index.json"
output_file = f"{data_path}/imdb_samples.txt"

# 加载数据
print("加载数据...")
with np.load(npz_file, allow_pickle=True) as f:
    x_train = f['x_train']   # 训练集评论序列（列表的数组）
    y_train = f['y_train']   # 训练集标签：0 负面，1 正面
    x_test = f['x_test']     # 测试集评论序列
    y_test = f['y_test']     # 测试集标签

with open(word_index_file, 'r') as f:
    word_index = json.load(f)   # 单词到索引的映射（索引从1开始）

# 构建反向索引（整数 -> 单词）
# 注意：IMDB 数据集保留 0、1、2 分别用于填充、序列开始、未知词
# 实际单词索引从 3 开始，因此需要将 word_index 中的值加上 3 来匹配序列中的数字
idx_to_word = {}
idx_to_word[0] = '<PAD>'
idx_to_word[1] = '<START>'
idx_to_word[2] = '<UNK>'
for word, idx in word_index.items():
    idx_to_word[idx + 3] = word   # 序列中的数字 = word_index[word] + 3

# 将整数序列转换为文本
def decode_review(sequence):
    """将整数序列解码为可读文本"""
    words = []
    for num in sequence:
        if num in idx_to_word:
            words.append(idx_to_word[num])
        else:
            words.append('<UNK>')   # 安全处理，理论上不会出现
    return ' '.join(words)

# 写入文件
print(f"写入样本到 {output_file} ...")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("训练集样本（前20条）\n")
    f.write("=" * 50 + "\n")
    for i in range(min(20, len(x_train))):
        label = "正面" if y_train[i] == 1 else "负面"
        text = decode_review(x_train[i])
        f.write(f"样本 {i+1} | 标签: {label}\n")
        f.write(f"文本: {text}\n\n")

    f.write("\n测试集样本（前20条）\n")
    f.write("=" * 50 + "\n")
    for i in range(min(20, len(x_test))):
        label = "正面" if y_test[i] == 1 else "负面"
        text = decode_review(x_test[i])
        f.write(f"样本 {i+1} | 标签: {label}\n")
        f.write(f"文本: {text}\n\n")

print("完成！")