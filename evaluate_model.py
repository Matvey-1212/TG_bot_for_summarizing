import os
import json
import torch
from datasets import load_dataset
import evaluate
from app.core.config import config
from app.models.model import TextModel


model = TextModel(config.MODEL_NAME, '', config.MODELS_CONFIG, config.NEWS_CLASS_DECODER, False)
dataset = load_dataset("IlyaGusev/gazeta")["test"]
texts = dataset["text"]
references = dataset["summary"]
print(f'Start inference')
predictions, _ = model.predict(texts, show_progress=True)

print(f'Start evaluating {config.MODEL_NAME}')
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=predictions, references=references)
# print("rouge", rouge_result)

meteor = evaluate.load("meteor")
meteor_result = meteor.compute(predictions=predictions, references=references)

bertscore = evaluate.load("bertscore")
bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="ru")
# print("BERTScore", bertscore_result)

avg_precision = sum(bertscore_result["precision"]) / len(bertscore_result["precision"])
avg_recall = sum(bertscore_result["recall"]) / len(bertscore_result["recall"])
avg_f1 = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])

final_results = {
    "model_name": config.MODEL_NAME,
    "meteor":meteor_result,
    "rouge": rouge_result,
    "bertscore": {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "hashcode": bertscore_result.get("hashcode", None)
    }
}

output_path = os.path.join(os.getcwd(), 'metrics')
output_file = os.path.join(output_path, f"{config.MODEL_NAME.replace('/','-')}_evaluation_results.json")
os.makedirs(output_path, exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4, ensure_ascii=False)

print(f"Results saved to {output_file}")
