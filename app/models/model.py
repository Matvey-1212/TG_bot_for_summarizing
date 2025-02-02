from transformers import AutoTokenizer, T5ForConditionalGeneration
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
        
    def predict(self, texts: str):
        keys = texts.keys()
        sequences = texts.values()
        
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
        for i in range(len(batch_input_sequences)):
            prediction = self.model.predict(batch_input_sequences[i])
            all_summary.extend(prediction)
        
        summary_dict = {}
        for key, val in zip(keys, all_summary):
            summary_dict[key] = val
        
        return summary_dict

model = TextModel(config.MODEL_NAME, config.MODELS_CONFIG)

if __name__ == "__main__":
    test_text = {
        # 'test1':"Промежуточный отчет о ходе работы должен охватывать не менее 50% работы, которая будет завершена к защите проекта. Отчет должен включать репозиторий на GitHub с вашим кодом, где ваш непрерывный прогресс должен быть явно виден (если вы участвуете в соревновании в команде, вклад всех участников должен быть явно виден через коммиты). Также необходимо предоставить краткий сопроводительный документ (например, текст или слайды с комментариями), описывающий, что вы сделали и что планируете сделать до завершения проекта. Отчет будет оцениваться в бинарной форме.",
        # 'test2':'На этой неделе администрация Дональда Трампа предложила госслужащим федерального уровня уволиться в обмен на выплату зарплаты до конца сентября. Как сообщает The Washington Post, среди первых получателей оказались и федеральные пожарные Калифорнии, которые уже почти месяц борются с самыми масштабными лесными пожарами в истории штата.',
        'test3':'Более 61 тысячи человек стали жертвами израильской военной операции в Секторе Газа — власти Палестины.'
                 }
    print(model.predict(test_text))
