# TG_bot_for_summarizing

## Overview
This branch implements a simple Machine Learning API using FastAPI. The primary goal is to enable model inference using Hugging Face models with an easy-to-use interface.

# mtv Tasks:
### 1 part
1. A basic ML API implemented with FastAPI.
2. Model inference using Hugging Face models.
3. A wrapper around the models for better usability.
4. Logging and configuration setup.
5. The baseline model is Mbart, fine-tuned on Russian news for text summarization.

### 2 part
1. finetune Mistral7b (**colab - too long, kuggle - can't start training, hpc - v100 doesn't support 4-8bit quantization**)
2. finetune mBart/GPT2
3. collect metrics
4. attempt to convert models pt->onnx->trt (**failure, onnx doesn' support decoder over 2gb**)
5. attempt to convert models to trt engine via TensorRT-llm (**failure, trt-llm doecn't support mbart/gpt2 architecture**)
6. convert gpt2 to GGUF, attempt to use llama.cpp as backend (**failure, llama.cpp doesn't support gpt2 architecture**)
7. use gigachat to markup gazeta valid set (for classification)
8. add classification model to inference
9. implementation of the database interaction code from ml
10. wrap in docker compose

