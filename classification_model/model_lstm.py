import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# 下载nltk数据包
nltk.download('punkt')

# 1. 读取CSV文件
data = pd.read_csv("lenta_news_correct.csv", encoding='gbk')


# 2. 检查数据的前几行
print(data.head())

# 3. 数据预处理
data = data.dropna(subset=['Текст', 'Ответ'])

# 4. 文本和标签
X = data['Текст']  # 文本列
y = data['Ответ']  # 分类标签列

# 5. 标签编码 (将分类标签转换为数字)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 6. 文本向量化（Tokenization）
max_vocab_size = 10000

# 创建一个词汇表
counter = Counter()
for text in X:
    counter.update(word_tokenize(text.lower()))  # 使用nltk进行分词

# 选择最常见的词汇
vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(max_vocab_size))}
vocab['<PAD>'] = 0  # 为填充添加一个索引
vocab['<UNK>'] = 1  # 为未知词添加一个索引

# 7. 文本转换为索引
def text_to_sequence(text):
    return [vocab.get(word, vocab['<UNK>']) for word in word_tokenize(text.lower())]

X_sequences = [text_to_sequence(text) for text in X]

# 8. 填充序列（确保每个序列的长度一致）
max_len = 150
X_pad = [seq[:max_len] if len(seq) > max_len else seq + [vocab['<PAD>']] * (max_len - len(seq)) for seq in X_sequences]

# 9. 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_encoded, test_size=0.2, random_state=42)

# 10. 自定义数据集类
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# 11. 创建DataLoader
train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 12. 定义神经网络模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, max_len):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.max_len = max_len
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(x)
        x = self.dropout(hn[-1])  # 只使用最后一个LSTM的输出
        x = self.fc(x)
        return x

# 13. 初始化模型、损失函数和优化器
model = LSTMClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=128, output_dim=len(label_encoder.classes_),
                       max_len=max_len)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 14. 训练模型
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")

# 15. 评估模型
# 15. 评估模型
def evaluate(model, test_loader):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    print("分类报告:")
    # 修正：确保labels参数的数量与y_true的类别一致
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, labels=range(len(label_encoder.classes_))))
    return y_pred, y_true


# 16. 训练和评估
for epoch in range(1, 2):  # 训练1个epoch
    train(model, train_loader, criterion, optimizer, epoch)

# 17. 测试集评估
y_pred, y_true = evaluate(model, test_loader)

# 18. 模型准确性
accuracy = (torch.tensor(y_pred) == torch.tensor(y_true)).float().mean()
print(f"模型准确性: {accuracy:.4f}")
