from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import torch
from app.core.config import config
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

class TextModel:
    def __init__(self, config):
        model_name = config.MODEL_NAME
        self.max_input = config.MAX_INPUT
        self.batch_size = config.BATCH_SIZE
        self.task_prefix = "" 
        # model_name = 'UrukHan/t5-russian-summarization'
        # self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # model_name = "IlyaGusev/rut5_base_sum_gazeta"
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name, force_download=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, force_download=False)

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
            encoded = self.tokenizer(
                [self.task_prefix + sequence for sequence in batch_input_sequences[i]],
                padding="longest",
                max_length=self.max_input,
                truncation=True,
                return_tensors="pt",
                )["input_ids"]
            
            predicts = self.model.generate(encoded, no_repeat_ngram_size=4, max_new_tokens=100) 
            summary = self.tokenizer.batch_decode(predicts, skip_special_tokens=True) 
            all_summary.extend(summary)
        
        summary_dict = {}
        for key, val in zip(keys, all_summary):
            summary_dict[key] = val
        
        return summary_dict

model = TextModel(config)

if __name__ == "__main__":
    test_text = {
        # 'test1':"Промежуточный отчет о ходе работы должен охватывать не менее 50% работы, которая будет завершена к защите проекта. Отчет должен включать репозиторий на GitHub с вашим кодом, где ваш непрерывный прогресс должен быть явно виден (если вы участвуете в соревновании в команде, вклад всех участников должен быть явно виден через коммиты). Также необходимо предоставить краткий сопроводительный документ (например, текст или слайды с комментариями), описывающий, что вы сделали и что планируете сделать до завершения проекта. Отчет будет оцениваться в бинарной форме.",
        # 'test2':'На этой неделе администрация Дональда Трампа предложила госслужащим федерального уровня уволиться в обмен на выплату зарплаты до конца сентября. Как сообщает The Washington Post, среди первых получателей оказались и федеральные пожарные Калифорнии, которые уже почти месяц борются с самыми масштабными лесными пожарами в истории штата.',
        'test3':'Более 61 тысячи человек стали жертвами израильской военной операции в Секторе Газа — власти Палестины.'
                 }
    print(model.predict(test_text))
