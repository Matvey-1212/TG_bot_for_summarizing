from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载保存的最佳模型和tokenizer
model_path = "./best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 使用与训练时相同的标签映射
labels = ['Здоровье и медицина', 'Культура и развлечения', 'Локальные новости', 'Метеорология',
          'Образование', 'Общество', 'Политика', 'Происшествия и криминал', 'Спорт',
          'Технологии и наука', 'Экономика и бизнес']
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    return predictions

# 示例文本
texts = [
    "В результате атаки беспилотного летательного аппарата (БПЛА) Вооруженных сил Украины (ВСУ) Белгородской области был ранен мужчина. Об этом сообщил глава российского региона Вячеслав Гладков в своем Telegram-канале",
    "?Ливерпуль? на своем поле обыграл французский ?Лилль? в матче седьмого тура общего этапа Лиги чемпионов. Об этом сообщает корреспондент ?Ленты.ру?"
]

predictions = predict(texts)
predicted_labels = [id2label[pred.item()] for pred in predictions]
print(predicted_labels)
