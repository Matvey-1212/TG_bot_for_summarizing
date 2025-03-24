# TG_bot_for_summarizing

## Overview
This branch implements a simple Machine Learning API using FastAPI. The primary goal is to enable model inference using Hugging Face models with an easy-to-use interface.

## Features
1. A basic ML API implemented with FastAPI.
2. Model inference using Hugging Face models.
3. A wrapper around the models for better usability.
4. Logging and configuration setup.
5. The baseline model is Mbart, fine-tuned on Russian news for text summarization.

## Plans
1. Conduct manual testing of the model on a dataset of Telegram news.
2. Fine-tune LLaMA 8B using QLoRA.
3. Rent a GPU server for model deployment.
