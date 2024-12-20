from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import jsonlines
import argparse
import os
from pathlib import Path

def setup_model(model_name):
    """Setup model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # trust_remote_code=True, # Qwenで必須
        token=os.environ.get("HUGGING_FACE_TOKEN")
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=os.environ.get("HUGGING_FACE_TOKEN")
    )
    return model, tokenizer

def generate_text(model, tokenizer, input_text):
    """Generate text using the model"""
    messages = [
        {"role": "system", "content": "あなたは役立つアシスタントです。ユーザーの質問に回答し、指示に従ってください。"},
        {"role": "user", "content": input_text}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    seed = 42
    torch.manual_seed(seed)

    tokens = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    output_tokens = tokens[0][model_inputs.input_ids.shape[1]:]
    return tokenizer.decode(output_tokens, skip_special_tokens=True)

def process_dataset(model, tokenizer, input_file, output_file):
    """Process the dataset and save predictions"""
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for i, obj in enumerate(reader):
            prompt = obj['input_text']
            generated_text = generate_text(model, tokenizer, prompt)
            writer.write({"pred": generated_text})

            print("========================================")
            print(f"Q. {prompt}")
            print(f"A. {generated_text}")
            print()
            
            if i % 10 == 0:
                torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Run inference on a dataset')
    parser.add_argument('--model', default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8", help='Model name or path')
    parser.add_argument('--input', required=True, help='Input JSONL file path')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model {args.model}...")
    model, tokenizer = setup_model(args.model)
    
    print(f"Processing dataset from {args.input}...")
    process_dataset(model, tokenizer, args.input, args.output)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()