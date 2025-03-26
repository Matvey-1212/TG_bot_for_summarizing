from functools import lru_cache
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import torch
from app.core.config import config
from app.models.models_interfaces import HUG_model, HUG_pipeline_model_classification, HUG_fake_news
from app.core.logging import logger


class TextModel:
    def __init__(self, 
                 name, 
                 config,
                 class_name=None,  
                 class_decoder=None, 
                 use_classifier=False,
                 fake_news_name=None,
                 use_fake_news=False,
                 ):
        local_config = config['models'].get(name, None)
        if local_config is None:
            logger.error(f"Cant find model name {name} in config")
            raise Exception(f"Cant find model name {name} in config")
        
        self.model = HUG_model(local_config)
        self.batch_size = local_config['batch_size']

        self.use_classifier = use_classifier
        if self.use_classifier:
            class_local_config = config['classifier'].get(class_name, None)
            if class_local_config is not None:
                self.class_model = HUG_pipeline_model_classification(class_local_config, class_decoder)
            else:
                self.use_classifier = False
             
        self.use_fake_news = use_fake_news   
        if self.use_fake_news:
            class_local_config = config['fake_news'].get(fake_news_name, None)
            if class_local_config is not None:
                self.fake_news_model = HUG_fake_news(class_local_config)
            else:
                self.use_classifier = False
        
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
        all_classes = []
        all_fake_news_prop = []
        iterator = range(len(batch_input_sequences))
        if show_progress:
            iterator = tqdm(iterator, desc="Processing batches", unit="batch")
        
        for i in iterator:
            prediction = self.model.predict(batch_input_sequences[i])
            all_summary.extend(prediction)
            
            if self.use_classifier:
                class_prediction = self.class_model.predict(prediction)
                all_classes.extend(class_prediction)
            else:
                all_classes.extend([0] * len(prediction))
                
            if self.use_fake_news:
                fake_news_prediction = self.fake_news_model.predict(prediction)
                all_fake_news_prop.extend(fake_news_prediction)
            else:
                all_fake_news_prop.extend([0] * len(batch_input_sequences[i]))
        
        return all_summary, all_classes, all_fake_news_prop
        

@lru_cache(maxsize=1)
def get_model():
    model = TextModel(config.MODEL_NAME, 
                      config.MODELS_CONFIG, 
                      config.CLASSIFICATION_MODEL_NAME, 
                      config.NEWS_CLASS_DECODER, 
                      config.USE_CLASSIFIER,
                      config.FAKE_NEWS_MODEL_NAME,
                      config.USE_FAKE_NEWS
                      )
    return model