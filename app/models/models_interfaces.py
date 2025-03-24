from abc import ABC, abstractmethod
from typing import Dict, List
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from app.core.logging import logger

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast, AutoModelForCausalLM
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
    elif name == 'T5ForConditionalGeneration':
        return T5ForConditionalGeneration
    elif name == 'AutoModelForCausalLM':
        return AutoModelForCausalLM
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
        if config['tokenizer_name'] == 'AutoTokenizer':
            self.tokenizer.padding_side = self.config['padding_side']
            
        self.model = get_model_class(config['model_class_name']).from_pretrained(config['model_name_path'])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device: {self.device}")
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model.half()
        self.model.eval()
        

    def predict(self, text: List[str]):
        input_str = [self.task_prefix + sequence for sequence in text]

        # [logger.debug(f"OUTPUT: {sequence}") for sequence in input_str]
        encoded = self.tokenizer(
            input_str,
            padding=self.config['tokenizer_padding'],
            max_length=self.config['max_input'],
            truncation=self.config['tokenizer_truncation'],
            add_special_tokens=self.config.get('add_special_tokens',True), 
            # padding_side=self.config.get('padding_side', None),
            return_tensors="pt",
            )["input_ids"]
        encoded = encoded.to(self.device)

        with torch.no_grad():
            predicts = self.model.generate(
                encoded, 
                do_sample=self.config.get('do_sample', False),
                num_beams=self.config['num_beams'],
                no_repeat_ngram_size=self.config['no_repeat_ngram_size'],
                max_new_tokens=self.config['max_new_tokens'],
                early_stopping=self.config['early_stopping'],
                top_k=self.config.get('top_k', None)
                ) 
        summary = self.tokenizer.batch_decode(predicts, skip_special_tokens=True) 
        return summary
        
