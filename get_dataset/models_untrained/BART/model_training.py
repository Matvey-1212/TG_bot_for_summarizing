from transformers import TrainingArguments, Trainer
from untrained_bart_model import model
from data_preparation import main

training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=4,   # 每个设备的训练批次大小
    per_device_eval_batch_size=4,    # 每个设备的评估批次大小
    warmup_steps=500,                # 热身步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,
    evaluation_strategy="steps",     # 评估策略
    eval_steps=50,
    save_steps=1000,
    save_total_limit=2,
)

train_dataset, val_dataset, _ = main()

trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=val_dataset,            # 验证数据集
)

trainer.train()