### Model Training
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# Define model name
model_name = "microsoft/DialoGPT-medium"

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    push_to_hub=False,
    report_to="none",
    remove_unused_columns=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./trained_chatbot")
tokenizer.save_pretrained("./trained_chatbot")
