import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer for QA (BART or T5)
model_name = "./trained_chatbot"  # Path to your fine-tuned model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the chatbot response function for answering questions
def chatbot_response(user_input):
    # Preprocess and tokenize the input text
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    # Generate the response using the model
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_length=50,  # Limit the response length
            num_return_sequences=1,
            temperature=0.7,  # Control randomness
            top_p=0.92,       # Use top-p sampling
            repetition_penalty=1.2, 
            do_sample=True,
            early_stopping=True
        )
    
    # Decode and return the answer (skip special tokens)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    return reply

# Create Gradio interface
iface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(label="Ask a question"),  # Label input
    outputs=gr.Textbox(label="Chatbot response"),  # Label output
    title="Education Chatbot",
    description="Ask me any educational question!",
    theme="compact",  # Optional: set theme to compact
)

# Launch Gradio app
if __name__ == "__main__":
    iface.launch(share=True)
