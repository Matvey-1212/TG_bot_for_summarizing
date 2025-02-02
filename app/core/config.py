import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    APP_PORT = int(os.getenv("APP_PORT", 8000))
    DEBUG = bool(os.getenv("DEBUG", True))
    MODEL_NAME = str(os.getenv("MODEL_NAME", 'IlyaGusev/rut5_base_sum_gazeta'))
    MAX_INPUT = int(os.getenv("MAX_INPUT", 512))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))

config = Config()
