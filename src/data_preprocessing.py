import torch
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import pandas as pd
import json

# Verify and Load SQuAD v2 dataset
def load_and_verify_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            print("JSON file loaded successfully!")
            return data
        except json.JSONDecodeError as e:
            print(f"JSON format error: {e}")
            return None

squad_data = load_and_verify_json("train-v2.0.json")
if squad_data is None:
    raise ValueError("Invalid JSON file. Please check and re-upload the dataset.")

# Load dataset
dataset = load_dataset("json", data_files={"train": "train-v2.0.json"}, field="data")

# Extract context and questions
def prepare_data(dataset):
    contexts, questions = [], []
    for entry in dataset["train"]:
        if "paragraphs" in entry:
            for paragraph in entry["paragraphs"]:
                if "qas" in paragraph:
                    for qa in paragraph["qas"]:
                        if "context" in paragraph and "question" in qa:
                            contexts.append(paragraph["context"])
                            questions.append(qa["question"])
    return {"context": contexts, "questions": questions}

data = prepare_data(dataset)

# Ensure equal lengths of context and questions
min_length = min(len(data["context"]), len(data["questions"]))
data["context"] = data["context"][:min_length]
data["questions"] = data["questions"][:min_length]

# Convert to dataset
dataset = Dataset.from_dict(data)

# Save dataset as CSV for download
df = pd.DataFrame(data)
df.to_csv("squad_data.csv", index=False)

# Load tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token=AutoTokenizer.from_pretrained(model_name).eos_token)

def preprocess_function(examples):
    # Tokenize context and question with truncation and padding to max_length
    inputs = tokenizer(
        examples["context"], 
        examples["questions"], 
        truncation=True, 
        padding="max_length",  # Pad to max length (512 tokens)
        max_length=512  # Limit to 512 tokens
    )
    return inputs

# Verify columns before calling remove_columns
print("Dataset columns before tokenization:", dataset.column_names)

# Tokenize dataset and remove original context and questions
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["context", "questions"])

# Train-Test Split
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]
