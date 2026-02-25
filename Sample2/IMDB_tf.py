"""
    自然语言处理任务：对IMDB电影评论进行情感分析，判断正负面倾向，并提取关键主题词。
针对IMDB基于TensorFlow 2.9的完整情感分析解决方案。代码包含以下模块：
    1.数据加载：从本地IMDB路径读取所有评论，整理为DataFrame。
    2.文本预处理：清洗文本（去除HTML标签、非字母字符）、分词、构建词汇表。
    3.构建词向量模型：使用Keras的Embedding层训练词嵌入。
    4.训练情感分类模型：搭建一个简单的双向LSTM网络进行分类。
    5.可视化情感分布：绘制训练集和测试集中正负样本的比例，以及模型预测结果的分布。
    6.提取高频关键词：分别统计正面和负面评论中的高频词，用词云和条形图展示。
    7.额外功能：使用TF-IDF提取每个类别下的主题词。
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# 设置TensorFlow随机种子以保证可重复性
tf.random.set_seed(42)

# 设置matplotlib正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题

# 配置参数
DATA_PATH = r'E:\Download\aclImdb'
MAX_VOCAB_SIZE = 20000   # 词汇表最大数量
MAX_SEQUENCE_LENGTH = 200  # 每条评论保留的最大单词数
EMBEDDING_DIM = 100       # 词向量维度
BATCH_SIZE = 128
EPOCHS = 10
VALIDATION_SPLIT = 0.2    # 从训练集中划分验证集的比例

# -------------------- 1. 数据加载 --------------------
def load_imdb_data(path):
    """从IMDB文件夹加载数据，返回DataFrame包含text和label"""
    texts = []
    labels = []
    for split in ['train', 'test']:
        for label, sentiment in [(1, 'pos'), (0, 'neg')]:
            folder = os.path.join(path, split, sentiment)
            if not os.path.exists(folder):
                continue
            for filename in os.listdir(folder):
                if filename.endswith('.txt'):
                    with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                    texts.append(text)
                    labels.append(label)
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df

print("正在加载数据...")
df = load_imdb_data(DATA_PATH)
print(f"数据加载完成，总样本数：{len(df)}")
print(df['label'].value_counts())

# 分割训练集和测试集（根据文件名已经分好，我们可以通过重新索引来区分）
# 由于数据加载是按顺序的：先train/pos, train/neg, test/pos, test/neg
# 所以前25000条是训练集（pos+neg），后25000条是测试集
train_df = df.iloc[:25000].copy()
test_df = df.iloc[25000:].copy()
print(f"训练集大小：{len(train_df)}，测试集大小：{len(test_df)}")

# -------------------- 2. 文本清洗与分词 --------------------
def clean_text(text):
    """清洗文本：去除HTML标签、非字母字符，转换为小写"""
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 保留字母和空格，去除数字和标点
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 转为小写
    text = text.lower()
    # 合并多个空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("正在清洗文本...")
train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)

# 使用Tokenizer进行分词并构建词汇表
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['clean_text'])

# 将文本转换为序列
train_sequences = tokenizer.texts_to_sequences(train_df['clean_text'])
test_sequences = tokenizer.texts_to_sequences(test_df['clean_text'])

# 填充/截断序列到固定长度
train_padded = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# 准备标签
train_labels = np.array(train_df['label'])
test_labels = np.array(test_df['label'])

print(f"训练数据形状：{train_padded.shape}，测试数据形状：{test_padded.shape}")

# -------------------- 3. 构建词向量模型与情感分类模型 --------------------
# 构建模型：嵌入层 + 双向LSTM + 全连接
model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    Bidirectional(LSTM(64, dropout=0.2, return_sequences=False)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')   # 二分类输出
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 早停回调
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("开始训练模型...")
history = model.fit(
    train_padded, train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stop],
    verbose=1
)

# 在测试集上评估
test_loss, test_acc = model.evaluate(test_padded, test_labels, verbose=0)
print(f"测试集准确率：{test_acc:.4f}")

# -------------------- 4. 可视化情感分布 --------------------
# 训练过程中准确率和损失曲线
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    # 准确率
    axes[0].plot(history.history['accuracy'], label='train_acc')
    axes[0].plot(history.history['val_accuracy'], label='val_acc')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    # 损失
    axes[1].plot(history.history['loss'], label='train_loss')
    axes[1].plot(history.history['val_loss'], label='val_loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history)

# 测试集上预测概率分布
test_pred_proba = model.predict(test_padded).flatten()
test_pred = (test_pred_proba > 0.5).astype(int)

# 绘制真实标签和预测标签的分布
fig, axes = plt.subplots(1, 2, figsize=(12,4))
sns.countplot(x=test_labels, ax=axes[0])
axes[0].set_title('True Label Distribution (Test Set)')
axes[0].set_xticklabels(['Negative', 'Positive'])
sns.countplot(x=test_pred, ax=axes[1])
axes[1].set_title('Predicted Label Distribution (Test Set)')
axes[1].set_xticklabels(['Negative', 'Positive'])
plt.tight_layout()
plt.savefig('label_distribution.png')
plt.show()

# 预测概率的直方图
plt.figure(figsize=(8,5))
plt.hist(test_pred_proba, bins=30, alpha=0.7, color='blue')
plt.title('Distribution of Predicted Probabilities (Test Set)')
plt.xlabel('Probability of Positive Class')
plt.ylabel('Frequency')
plt.savefig('probability_distribution.png')
plt.show()

# -------------------- 5. 提取高频关键词与主题 --------------------
# 获取正面和负面评论的文本
pos_texts = train_df[train_df['label']==1]['clean_text'].tolist()
neg_texts = train_df[train_df['label']==0]['clean_text'].tolist()

# 定义函数统计词频
def get_top_words(texts, tokenizer, top_n=20):
    # 将文本转换为单词列表（使用tokenizer的索引反转）
    word_index = tokenizer.word_index
    index_word = {v:k for k,v in word_index.items()}
    # 统计词频
    all_words = []
    for seq in tokenizer.texts_to_sequences(texts):
        for idx in seq:
            if idx < MAX_VOCAB_SIZE:  # 只考虑词汇表内的词
                all_words.append(index_word.get(idx, '<OOV>'))
    counter = Counter(all_words)
    return counter.most_common(top_n)

# 获取正面和负面高频词
top_pos = get_top_words(pos_texts, tokenizer, top_n=20)
top_neg = get_top_words(neg_texts, tokenizer, top_n=20)

print("正面评论高频词：")
for word, count in top_pos:
    print(f"{word}: {count}")

print("\n负面评论高频词：")
for word, count in top_neg:
    print(f"{word}: {count}")

# 绘制条形图
def plot_top_words(top_words, title, color):
    words, counts = zip(*top_words)
    plt.figure(figsize=(10,6))
    plt.barh(words, counts, color=color)
    plt.xlabel('Frequency')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()

plot_top_words(top_pos, 'Top Words in Positive Reviews', 'green')
plot_top_words(top_neg, 'Top Words in Negative Reviews', 'red')

# 生成词云
def generate_wordcloud(texts, title):
    all_text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()

generate_wordcloud(pos_texts, 'Positive Reviews Word Cloud')
generate_wordcloud(neg_texts, 'Negative Reviews Word Cloud')

# -------------------- 6. 额外：使用TF-IDF提取主题词 --------------------
# 从训练集中分别提取正负面评论的TF-IDF主题词
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
# 合并正负面评论用于训练TF-IDF
all_texts = pos_texts + neg_texts
tfidf_vectorizer.fit(all_texts)

# 获取特征名（词汇）
feature_names = tfidf_vectorizer.get_feature_names_out()

# 计算每个类别的平均TF-IDF得分
def get_top_tfidf_words(texts, vectorizer, top_n=20):
    # 将文本转换为TF-IDF矩阵
    tfidf_matrix = vectorizer.transform(texts)
    # 计算每个词的平均得分
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    # 获取得分最高的词索引
    top_indices = mean_tfidf.argsort()[-top_n:][::-1]
    top_words = [(feature_names[i], mean_tfidf[i]) for i in top_indices]
    return top_words

top_pos_tfidf = get_top_tfidf_words(pos_texts, tfidf_vectorizer, top_n=20)
top_neg_tfidf = get_top_tfidf_words(neg_texts, tfidf_vectorizer, top_n=20)

print("\n正面评论TF-IDF主题词：")
for word, score in top_pos_tfidf:
    print(f"{word}: {score:.4f}")

print("\n负面评论TF-IDF主题词：")
for word, score in top_neg_tfidf:
    print(f"{word}: {score:.4f}")

# 绘制TF-IDF主题词条形图
def plot_tfidf_words(top_words, title, color):
    words, scores = zip(*top_words)
    plt.figure(figsize=(10,6))
    plt.barh(words, scores, color=color)
    plt.xlabel('Average TF-IDF Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}_tfidf.png')
    plt.show()

plot_tfidf_words(top_pos_tfidf, 'Top TF-IDF Words in Positive Reviews', 'green')
plot_tfidf_words(top_neg_tfidf, 'Top TF-IDF Words in Negative Reviews', 'red')

print("所有任务完成！")


