from functools import lru_cache
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import torch
from app.core.config import config
from app.models.models_interfaces import HUG_model
from app.core.logging import logger


class TextModel:
    def __init__(self, name, config):
        local_config = config.get(name, None)
        if local_config is None:
            logger.error(f"Cant find model name {name} in config")
            raise Exception(f"Cant find model name {name} in config")
        
        self.model = HUG_model(local_config)
        self.batch_size = local_config['batch_size']
        
    def predict(self, texts: str, show_progress: bool = False):
        # keys = texts.keys()
        sequences = texts
        
        batch_input_sequences = []
        input_sequences = []
        for text in sequences:
            if len(input_sequences) <= self.batch_size:
                input_sequences.append(text)
            else:
                batch_input_sequences.append(input_sequences)
                input_sequences = [text]
        batch_input_sequences.append(input_sequences)   
        
        all_summary = []
        iterator = range(len(batch_input_sequences))
        if show_progress:
            iterator = tqdm(iterator, desc="Processing batches", unit="batch")
        
        for i in iterator:
            prediction = self.model.predict(batch_input_sequences[i])
            all_summary.extend(prediction)
        
        return all_summary
        
        # summary_dict = {}
        # for key, val in zip(keys, all_summary):
        #     summary_dict[key] = val
        
        return all_summary

# model = TextModel(config.MODEL_NAME, config.MODELS_CONFIG)
@lru_cache(maxsize=1)
def get_model():
    model = TextModel(config.MODEL_NAME, config.MODELS_CONFIG)
    return model