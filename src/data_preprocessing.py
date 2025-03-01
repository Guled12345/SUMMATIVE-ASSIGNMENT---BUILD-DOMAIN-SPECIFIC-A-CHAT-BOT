### Data Processing
import torch
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

# Load dataset from CSV
df = pd.read_csv("/content/text_data_toc.csv")  # Update path if needed
print("Dataset loaded successfully!")
print("Dataset Columns:", df.columns)

# Rename columns to match expected names
df.rename(columns={"words": "context", "file": "questions"}, inplace=True)

# Ensure the required columns exist
required_columns = ["context", "questions"]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Dataset is missing required columns: {required_columns}")

# Convert columns to string type to ensure compatibility with tokenizer
df["context"] = df["context"].astype(str)
df["questions"] = df["questions"].astype(str)

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Save dataset as CSV for reference
df.to_csv("squad_data.csv", index=False)

# Load tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Tokenization function
def preprocess_function(examples):
    inputs = tokenizer(
        examples["context"],
        examples["questions"],
        truncation=True,
        padding="max_length",
        max_length=256,  # Reduce max_length to speed up processing
        return_tensors="pt",
        return_attention_mask=True
    )
    return inputs

# Print dataset columns before tokenization
print("Dataset columns before tokenization:", dataset.column_names)

# Tokenize dataset and remove original text columns
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["context", "questions"])

# Train-Test Split
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1, shuffle=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

print("Data processing completed successfully!")
