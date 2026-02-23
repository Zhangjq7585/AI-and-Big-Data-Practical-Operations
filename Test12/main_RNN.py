##使用 PyTorch 1.9 + torchtext 0.10.0 构建 RNN 文本分类模型（以 IMDB 情感分析为例）的核心代码


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# ---------------------- 1. 数据加载与预处理 ----------------------
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 分词器（英文）
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    """从迭代器中逐个产生经过分词的文本"""
    for label, line in data_iter:
        yield tokenizer(line)

# 加载训练、测试迭代器（torchtext 0.10.0 返回迭代器）
train_iter, test_iter = IMDB(split=('train', 'test'))

# 构建词汇表：使用训练集，设置特殊标记和最大词汇量
# vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<pad>', '<unk>'], max_tokens=25000)    #报错！
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<pad>', '<unk>'])
vocab.set_default_index(vocab['<unk>'])  # 未知词索引

# 文本 → 索引序列
def text_pipeline(text):
    return vocab(tokenizer(text))

# 标签 → 数值（1: pos, 0: neg）
def label_pipeline(label):
    return 1 if label == 'pos' else 0

# 自定义 Dataset，将迭代器数据转化为可索引的数据集
class IMDBDataset(Dataset):
    def __init__(self, data_iter):
        self.data = [(label_pipeline(label), torch.tensor(text_pipeline(text), dtype=torch.long))
                     for label, text in data_iter]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 重新获取迭代器（因为之前被消耗了）
train_iter, test_iter = IMDB(split=('train', 'test'))
train_dataset = IMDBDataset(train_iter)
test_dataset = IMDBDataset(test_iter)

# 自定义 collate 函数，对 batch 内序列进行填充
def collate_batch(batch):
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(label)
        text_list.append(text)
    labels = torch.tensor(label_list, dtype=torch.float32)
    texts = pad_sequence(text_list, batch_first=True, padding_value=vocab['<pad>'])
    return texts.to(device), labels.to(device)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# ---------------------- 2. 模型定义 ----------------------
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, pad_idx):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # 使用 LSTM（可换为 nn.RNN 或 nn.GRU）
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=False, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)   # 二分类输出1个值，配合BCEWithLogitsLoss

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)   # [batch_size, seq_len, embed_dim]
        # 取最后一个时间步的隐藏状态（此处使用LSTM的输出h_n）
        output, (hidden, cell) = self.rnn(embedded)
        # hidden: [num_layers, batch, hidden_dim] 取最后一层
        last_hidden = hidden[-1]       # [batch_size, hidden_dim]
        logits = self.fc(last_hidden)  # [batch_size, 1]
        return logits.squeeze(1)       # 去掉多余维度 [batch_size]

# 超参数
VOCAB_SIZE = len(vocab)
EMBED_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_CLASSES = 1    # 二分类
PAD_IDX = vocab['<pad>']

model = RNNClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, PAD_IDX).to(device)

# ---------------------- 3. 训练 ----------------------
criterion = nn.BCEWithLogitsLoss()   # 二分类，配合sigmoid
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for texts, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(texts)                 # [batch_size]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5   # 二分类阈值0.5
        correct += (preds == labels.bool()).sum().item()
        total += labels.size(0)

    return total_loss / len(data_loader), correct / total

# ---------------------- 4. 评估 ----------------------
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in data_loader:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)
    return total_loss / len(data_loader), correct / total

# 训练循环
EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f'Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
          f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')


