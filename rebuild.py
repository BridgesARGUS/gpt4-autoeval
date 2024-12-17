import re
import json
from pathlib import Path

def parse_qa_log(log_file):
    # Read the log file
    with open(log_file, 'r', encoding='utf-8') as f:
        log_text = f.read()
    
    # Split the log into QA pairs
    qa_pairs = log_text.split("========================================\n")[1:]
    
    # Parse each QA pair
    results = []
    for pair in qa_pairs:
        if not pair.strip():  # Skip empty pairs
            continue
            
        # Extract Q and A using regex
        q_match = re.search(r'Q\. (.*?)\nA\. (.*?)(?=\n={40}|\Z)', pair, re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
            answer = q_match.group(2).strip()
            
            # Create prediction entry
            pred_entry = {"pred": answer}
            results.append(pred_entry)
    
    return results

def write_jsonl(data, output_file):
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            json_str = json.dumps(entry, ensure_ascii=False)
            f.write(json_str + '\n')

def process_log_to_jsonl(input_file, output_file):
    print(f"Reading log from: {input_file}")
    results = parse_qa_log(input_file)
    print(f"Found {len(results)} QA pairs")
    
    write_jsonl(results, output_file)
    print(f"Wrote predictions to: {output_file}")
    
    # Verify a few entries
    print("\nVerifying first few entries...")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Only show first 3 entries
                break
            entry = json.loads(line)
            print(f"\nEntry {i+1}:")
            print(f"{entry['pred'][:100]}...")

if __name__ == "__main__":
    # Input and output file paths
    input_log = "/workspaces/gpt4-autoeval/origin.txt"    # ← ログファイルのパスを指定
    output_jsonl = "/workspaces/gpt4-autoeval/preds.jsonl"      # ← 出力先のパスを指定
    
    process_log_to_jsonl(input_log, output_jsonl)