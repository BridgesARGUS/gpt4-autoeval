import json

def calculate_grades(file_path):
    """
    Calculate the sum and average of grades from a JSONL file
    
    Parameters:
    file_path (str): Path to the JSONL file
    
    Returns:
    tuple: (total_sum, average)
    """
    total_sum = 0
    count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse each line as JSON
            data = json.loads(line)
            # Add the grade to total
            total_sum += data['grade']
            count += 1
    
    # Calculate average
    average = total_sum / count if count > 0 else 0
    
    return total_sum, average

# Example usage
if __name__ == '__main__':
    file_path = '../assets/elyza_tasks_100/Qwen2.5-7B-Instruct/result.jsonl'
    total, avg = calculate_grades(file_path)
    print(f'合計点: {total}')
    print(f'平均点: {avg:.2f}')