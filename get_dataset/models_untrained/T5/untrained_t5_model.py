import torch
from transformers import T5Config, T5ForConditionalGeneration

config = T5Config()

model = T5ForConditionalGeneration(config)

print(model)