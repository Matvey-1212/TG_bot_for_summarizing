import os
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PARAMS_PATH = BASE_DIR / "model_params.json"
NEWS_CLASS_DECODER_PATH = BASE_DIR / "app/core/news_class_config.json"

class Config:
    APP_PORT = int(os.getenv("APP_PORT", 8000))
    DEBUG = bool(os.getenv("DEBUG", True))
    MODEL_NAME = str(os.getenv("MODEL_NAME", 'IlyaGusev/rut5_base_sum_gazeta'))
    
    USE_CLASSIFIER = bool(os.getenv("USE_CLASSIFIER", False))
    CLASSIFICATION_MODEL_NAME = str(os.getenv("CLASSIFICATION_MODEL_NAME", ''))
    
    USE_FAKE_NEWS = bool(os.getenv("USE_FAKE_NEWS", False))
    FAKE_NEWS_MODEL_NAME = str(os.getenv("FAKE_NEWS_MODEL_NAME", ''))
    
    with open(MODEL_PARAMS_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    MODELS_CONFIG = json_data
    
    with open(NEWS_CLASS_DECODER_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    NEWS_CLASS_DECODER = json_data
     

config = Config()
