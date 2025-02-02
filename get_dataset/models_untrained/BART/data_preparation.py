import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def split_dataset(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    return train_data, val_data, test_data

def preprocess_function(examples, tokenizer):
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    file_path = 'tba_dataset.csv'
    data = load_data(file_path)
    train_data, val_data, test_data = split_dataset(data)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    train_dataset = train_data.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    val_dataset = val_data.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    test_dataset = test_data.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    main()