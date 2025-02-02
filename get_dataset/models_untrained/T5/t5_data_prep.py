import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return data

def split_datasets(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    return train_data, val_data, test_data

def preprocess(examples, tokenizer):
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    labels = tokenizer(examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    file_path = 'your_dataset.csv'
    data = load_dataset(file_path)
    train_data, val_data, test_data = split_datasets(data)

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    train_dataset = train_data.map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer})
    val_dataset = val_data.map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer})
    test_dataset = test_data.map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer})

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    main()