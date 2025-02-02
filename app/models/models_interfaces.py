from abc import ABC, abstractmethod
from typing import Dict, List
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from app.core.logging import logger

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast
from transformers import MBartForConditionalGeneration, MBartTokenizer

def get_tokenizer(name):
    if name == 'T5TokenizerFast':
        return T5TokenizerFast
    elif name == 'MBartTokenizer':
       return MBartTokenizer
    elif name == 'AutoTokenizer':
        return AutoTokenizer
    else:
        logger.error(f"Cant find suitable tokenizer type for {name}")
        raise Exception(f"Cant find suitable tokenizer type for {name}")
    
def get_model_class(name):
    if name == 'AutoModelForSeq2SeqLM':
        return AutoModelForSeq2SeqLM
    elif name == 'MBartForConditionalGeneration':
        return MBartForConditionalGeneration
    else:
        logger.error(f"Cant find suitable model class type for {name}")
        raise Exception(f"Cant find suitable model class type for {name}")

class ModelWrapper(ABC):
    @abstractmethod
    def predict(self, text: str):
        pass

class HUG_model(ModelWrapper):
    def __init__(self, config):
        print(config)
        self.config = config
        logger.debug(f"Model: {config['model_name_path']}, tokenizer: {config['tokenizer_name']}, model_class: {config['model_class_name']}")
        logger.debug(f'{self.config}')
        
        self.task_prefix = config['task_prefix']
        self.tokenizer = get_tokenizer(config['tokenizer_name']).from_pretrained(config['model_name_path'])
        self.model = get_model_class(config['model_class_name']).from_pretrained(config['model_name_path'])
        self.model.eval()
        

    def predict(self, text: List[str]):
        input = [self.task_prefix + sequence for sequence in text]
        encoded = self.tokenizer(
            input,
            padding=self.config['tokenizer_padding'],
            max_length=self.config['max_input'],
            truncation=self.config['tokenizer_truncation'],
            return_tensors="pt",
            )["input_ids"]
        
        predicts = self.model.generate(encoded, no_repeat_ngram_size=self.config['no_repeat_ngram_size'], max_new_tokens=self.config['max_new_tokens']) 
        summary = self.tokenizer.batch_decode(predicts, skip_special_tokens=True) 
        return summary
        
