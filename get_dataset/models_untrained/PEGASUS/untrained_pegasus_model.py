import torch
from transformers import PegasusConfig, PegasusForConditionalGeneration

config = PegasusConfig()

model = PegasusForConditionalGeneration(config)

print(model)