import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import jsonlines
import argparse
from pathlib import Path
import os

def setup_model(model_name):
    """Setup model and tokenizer with additional safeguards"""
    try:
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,  # Using slow tokenizer for better compatibility
            token=os.environ.get("HUGGING_FACE_TOKEN")
        )
        
        print(f"Loading model from {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Using float16 for better memory efficiency
            revision="main",  # Explicitly specify the main branch
            token=os.environ.get("HUGGING_FACE_TOKEN")
        )
        model.eval()  # Ensure model is in evaluation mode
        return model, tokenizer
    except Exception as e:
        print(f"Error during model setup: {str(e)}")
        raise

def generate_text(model, tokenizer, input_text):
    """Generate text using the model"""
    try:
        messages = [
            {"role": "system", "content": "あなたは役立つアシスタントです。ユーザーの質問に回答し、指示に従ってください。"},
            {"role": "user", "content": input_text}
        ]

        # Check if apply_chat_template is available
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to basic formatting
            prompt_text = f"System: あなたは役立つアシスタントです。ユーザーの質問に回答し、指示に従ってください。\nUser: {input_text}\nAssistant:"

        model_inputs = tokenizer(
            [prompt_text], 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Adjust based on model's context window
        ).to(model.device)

        # Set seed for reproducibility
        torch.manual_seed(42)
        
        with torch.inference_mode():
            tokens = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        output_tokens = tokens[0][model_inputs.input_ids.shape[1]:]
        return tokenizer.decode(output_tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        return f"Error generating response: {str(e)}"

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
                
                # Clear CUDA cache periodically
                if i % 5 == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Run inference on a dataset')
    parser.add_argument('--model', default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8", help='Model name or path')
    parser.add_argument('--input', required=True, help='Input JSONL file path')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup model and process dataset
    model, tokenizer = setup_model(args.model)
    process_dataset(model, tokenizer, args.input, args.output)

if __name__ == "__main__":
    main()