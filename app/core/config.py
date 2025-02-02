import os
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PARAMS_PATH = BASE_DIR / "model_params.json"

class Config:
    APP_PORT = int(os.getenv("APP_PORT", 8000))
    DEBUG = bool(os.getenv("DEBUG", True))
    MODEL_NAME = str(os.getenv("MODEL_NAME", 'IlyaGusev/rut5_base_sum_gazeta'))
    
    with open(MODEL_PARAMS_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    MODELS_CONFIG = json_data['models']
     

config = Config()
