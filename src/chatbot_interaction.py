### Chatbot Interaction
import gradio as gr
import torch

def chatbot_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        reply_ids = model.generate(
            **inputs, 
            max_length=30,  # Reduce length for faster response
            num_return_sequences=1, 
            temperature=0.7,  
            top_p=0.9,  
            repetition_penalty=1.2,
            do_sample=True,  
            early_stopping=True  # Stops early to speed up
        )

    return tokenizer.decode(reply_ids[0], skip_special_tokens=True)

# Create Gradio interface
iface = gr.Interface(
    fn=chatbot_response,
    inputs="text",
    outputs="text",
    title="Education Chatbot",
    description="Ask me any educational question!",
)

# Launch Gradio app
if __name__ == "__main__":
    iface.launch(share=True)

print("Dataset saved as squad_data.csv. You can download it from your environment.")
