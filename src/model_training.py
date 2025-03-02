import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from huggingface_hub import login
import os

# ✅ Disable WANDB logging (optional)
os.environ["WANDB_MODE"] = "disabled"

# ✅ Authenticate with Hugging Face token (Replace with your token if needed)
login("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")

# ✅ Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# ✅ Fix: Read dataset correctly (TSV → CSV)
original_file = "merged_question_answer_pairs.csv"  # Ensure this file is uploaded

# ✅ Read dataset as TSV (force tab separator & correct encoding)
df = pd.read_csv(original_file, sep="\t", encoding="utf-8-sig", engine="python")

# ✅ Debug: Check detected columns
print("\n📌 Detected Columns:", df.columns)

# ✅ Check if dataset is being read as a single column
if len(df.columns) == 1:
    print("\n❌ ERROR: Dataset is not properly split! Fixing column separation...")
    
    # ✅ Force Pandas to split columns correctly
    df = df.iloc[:, 0].str.split("\t", expand=True)

    # ✅ Rename columns manually
    df.columns = ["ArticleTitle", "Question", "Answer", "DifficultyFromQuestioner", "DifficultyFromAnswerer", "ArticleFile"]

# ✅ Drop missing values in `Question` or `Answer` columns
df.dropna(subset=["Question", "Answer"], inplace=True)

# ✅ Debug: Show first few rows
print("\n📌 First 5 Rows of Fixed Dataset:")
print(df.head())

# ✅ Save it as a proper CSV with commas
fixed_file = "fixed_dataset.csv"
df.to_csv(fixed_file, index=False)

print(f"\n✅ Successfully converted dataset: {fixed_file}")

# ✅ Reload dataset with correct column separation
dataset = load_dataset("csv", data_files=fixed_file, split="train")

# ✅ Debugging: Print dataset columns
print(f"\n📌 Available Dataset Columns: {dataset.column_names}")

# ✅ Fix column selection
question_column_name = "Question"
answer_column_name = "Answer"

if question_column_name not in dataset.column_names or answer_column_name not in dataset.column_names:
    raise KeyError(f"❌ ERROR: Dataset does not contain expected columns!\n"
                   f"✅ Available Columns: {dataset.column_names}")

print(f"✅ Detected Column Names: Question → {question_column_name}, Answer → {answer_column_name}")

# ✅ Split dataset into training & validation
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ✅ Load pre-trained model
model_name = "facebook/bart-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Tokenization function (Handle Missing Values Safely)
def process_data_for_training(dataset):
    def tokenize_function(examples):
        # ✅ Ensure no missing values before processing
        inputs = [text.strip() if text is not None else "" for text in examples[question_column_name]]
        targets = [text.strip() if text is not None else "" for text in examples[answer_column_name]]

        tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
        tokenized_targets = tokenizer(targets, padding="max_length", truncation=True, max_length=128)

        labels = tokenized_targets["input_ids"]
        labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels
        }

    return dataset.map(tokenize_function, batched=True)

# ✅ Process datasets
train_dataset = process_data_for_training(train_dataset)
eval_dataset = process_data_for_training(eval_dataset)

# ✅ Set up training arguments (optimized for Colab)
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True if torch.cuda.is_available() else False,  # Enable mixed precision if GPU is available
    dataloader_num_workers=2
)

# ✅ Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ✅ Start training
trainer.train()

# ✅ Save trained model & tokenizer
model.save_pretrained("./trained_chatbot")
tokenizer.save_pretrained("./trained_chatbot")

# ✅ Function to generate chatbot responses
def generate_response(input_text):
    input_text = input_text.strip()
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# ✅ Example Usage
user_input = "What areas did Beyonce compete in when she was growing up?"
response = generate_response(user_input)
print("🤖 Chatbot:", response)
