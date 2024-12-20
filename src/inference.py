import json
import jsonlines
import openai
from typing import List, Dict

# OpenAIのクライアントを設定
client = openai.OpenAI(
    api_key="your-api-key-here"  # ✋実際のAPIキーに置き換えてください
)

def format_prompt(input_text: str) -> List[Dict]:
    """
    OpenAIのチャットフォーマットに合わせてプロンプトを整形
    """
    messages = [
        {"role": "user", "content": input_text}
    ]
    return messages

def generate_text(messages: List[Dict]) -> str:
    """
    OpenAI GPT-4を使用してテキストを生成
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # または "gpt-4-turbo-preview" など
            messages=messages,
            max_tokens=1024,
            temperature=0.8,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        return ""

def process_dataset(input_file: str, output_file: str):
    """
    データセットを処理し、予測を生成して保存
    """
    try:
        with jsonlines.open(input_file) as reader, \
             jsonlines.open(output_file, mode='w') as writer:
            for obj in reader:
                prompt = obj['input_text']
                messages = format_prompt(prompt)
                generated_text = generate_text(messages)
                print(generated_text)
                writer.write({"pred": generated_text})
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    # データセットの処理を実行
    input_dataset = '../assets/elyza_tasks_100/gemma-2-27b-it-GPTQ/dataset.jsonl'
    output_predictions = '../assets/elyza_tasks_100/gemma-2-27b-it-GPTQ/preds.jsonl'
    process_dataset(input_dataset, output_predictions)