from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import jsonlines
import argparse
from pathlib import Path

def setup_model(model_name):
    """Setup model and tokenizer with Gemma-specific configurations"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    ).eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, input_text):
    """Generate text using Gemma's specific format"""
    messages = [
        {"role": "system", "content": "あなたは役立つアシスタントです。ユーザーの質問に回答し、指示に従ってください。"},
        {"role": "user", "content": input_text}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048  # Gemma context window
    ).to(model.device)

    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Gemma-specific generation parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def process_dataset(model, tokenizer, input_file, output_file):
    """Process the dataset and save predictions"""
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for i, obj in enumerate(reader):
            try:
                prompt = obj['input_text']
                generated_text = generate_text(model, tokenizer, prompt)
                writer.write({"pred": generated_text})
                
                print("========================================")
                print(f"Q. {prompt}")
                print(f"A. {generated_text}")
                print()
                
                # Clear CUDA cache more frequently for memory management
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                writer.write({"pred": "Error generating response"})

def main():
    parser = argparse.ArgumentParser(description='Run inference with Gemma 2 27B IT GPTQ')
    parser.add_argument('--model', default="shuyuej/gemma-2-27b-it-GPTQ", help='Model name or path')
    parser.add_argument('--input', required=True, help='Input JSONL file path')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Gemma model {args.model}...")
    model, tokenizer = setup_model(args.model)
    
    print(f"Processing dataset from {args.input}...")
    process_dataset(model, tokenizer, args.input, args.output)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()