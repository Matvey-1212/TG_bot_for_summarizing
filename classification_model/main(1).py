import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 读取数据
data = pd.read_csv("lenta_news_correct.csv", encoding='gbk')
labels = sorted(data["Ответ"].unique())
print(labels)
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
data["label"] = data["Ответ"].map(label2id)

# 划分训练集和测试集
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data["label"])

# 模型和分词器的初始化
model_name = "seara/rubert-tiny2-russian-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)


# 分词函数
def tokenize_function(examples):
    return tokenizer(examples["Текст"], truncation=True, padding="max_length", max_length=128)


# 构建数据集
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# 计算评估指标
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    # 如果是二分类问题，还可以计算AUC
    if len(np.unique(labels)) == 2:
        auc = roc_auc_score(labels, p.predictions[:, 1])  # 计算正类的AUC
    else:
        auc = None

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


# 训练配置
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 使用Trainer API进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 保存最好的模型
trainer.save_model("./best_model")
tokenizer.save_pretrained("./best_model")

# 评估训练集和测试集的表现
train_metrics = trainer.evaluate(train_dataset)
test_metrics = trainer.evaluate(test_dataset)

# 输出评估指标
print("Train Metrics:", train_metrics)
print("Test Metrics:", test_metrics)
