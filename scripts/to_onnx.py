from transformers import MBartForConditionalGeneration, MBartTokenizer, AutoTokenizer
import torch
import onnx
import os

model_name = "Kirili4ik/mbart_ruDialogSum"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "test text input"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

model.eval()
with torch.no_grad():
    cur_dir = os.getcwd()
    onnx_path = os.path.join(cur_dir, 'scripts/onnx_model')
    onnx_filename = os.path.join(onnx_path, "mbart_model.onnx")
    
    os.makedirs(onnx_path, exist_ok=True)
    
    torch.onnx.export(model,
                      (inputs["input_ids"],),
                      onnx_filename,
                      input_names=["input_ids"],
                      output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                                    "logits": {0: "batch_size", 1: "sequence_length"}},
                      opset_version=14)

