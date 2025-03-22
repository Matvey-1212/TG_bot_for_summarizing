from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import onnx

model_name = "Kirili4ik/mbart_ruDialogSum"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

input_text = "Translate English to German: How are you?"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

model.eval()
with torch.no_grad():
    onnx_filename = "t5_model.onnx"
    
    torch.onnx.export(model,
                      (inputs["input_ids"],),
                      onnx_filename,
                      input_names=["input_ids"],
                      output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                                    "logits": {0: "batch_size", 1: "sequence_length"}},
                      opset_version=12)

print(f"Модель успешно сохранена в формат ONNX: {onnx_filename}")
