## 数据集路径是E:\Download\imdb，包含imdb.npz和imdb_word_index.json。
## 需要写出数据导入、模型定义、训练、评估的核心代码。PyTorch版本1.9，Python 3.9

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

# -------------------- 1. 数据导入与预处理 --------------------
# 数据集路径
data_path = r"E:\Download\imdb"
npz_file = f"{data_path}/imdb.npz"
word_index_file = f"{data_path}/imdb_word_index.json"

# 加载 npz 文件（Keras 格式，包含 x_train, y_train, x_test, y_test）
with np.load(npz_file, allow_pickle=True) as f:
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']

# 加载单词索引（用于查看单词，可选）
with open(word_index_file, 'r') as f:
    word_index = json.load(f)

# 数据预处理：将评论序列填充/截断到相同长度
max_len = 200  # 选择最大序列长度
def pad_sequences(sequences, maxlen, dtype='int64'):
    """
    将序列列表填充/截断到固定长度，返回 numpy 数组。
    """
    padded = np.zeros((len(sequences), maxlen), dtype=dtype)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            # 截断（保留后 maxlen 个词，可根据需要调整）
            padded[i] = seq[:maxlen]
        else:
            padded[i, :len(seq)] = seq
    return padded

# 对训练和测试数据进行填充
X_train = pad_sequences(x_train, maxlen=max_len)
X_test = pad_sequences(x_test, maxlen=max_len)

# 标签转换为 float（二分类）
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # 形状 (N,1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# -------------------- 2. 定义 RNN 模型 --------------------
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # 索引 0 作为填充
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text shape: (batch_size, seq_len)
        embedded = self.embedding(text)          # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        # LSTM 输出: (batch_size, seq_len, hidden_dim), (h_n, c_n)
        output, (hidden, cell) = self.rnn(embedded)
        # 取最后一个时间步的隐藏状态 (LSTM 返回的 hidden 是最后一层所有层，取最后一层)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        last_hidden = hidden[-1, :, :]           # (batch_size, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)               # (batch_size, output_dim)
        return self.sigmoid(out)

# 超参数
vocab_size = len(word_index) + 3  # 加3是因为 Keras IMDB 索引从1开始，0是填充，1是起始，2是未知
embedding_dim = 100
hidden_dim = 128
output_dim = 1
num_layers = 2
dropout = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout).to(device)

# -------------------- 3. 训练设置 --------------------
criterion = nn.BCELoss()          # 二分类交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

# -------------------- 4. 训练循环 --------------------
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # 计算准确率
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

print("开始训练...")
train_model(model, train_loader, criterion, optimizer, epochs)

# -------------------- 5. 评估 --------------------
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

print("评估模型...")
evaluate_model(model, test_loader)