import torch
from transformers import BartConfig, BartForConditionalGeneration

config = BartConfig()

model = BartForConditionalGeneration(config)

print(model)