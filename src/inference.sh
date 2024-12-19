#!/bin/bash

# Exit on error
set -e

# Script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Starting inference setup and execution..."

# Install required packages if not already installed
if ! pip show optimum torch transformers accelerate auto-gptq > /dev/null; then
    echo "Installing dependencies..."
    pip install -U "optimum>=1.20.0"
    pip install torch transformers accelerate auto-gptq
else
    echo "Dependencies already installed, skipping installation..."
fi

# Function to run inference
run_inference() {
    local input_file=$1
    local output_file=$2
    local model_name=$3
    
    if [ -z "$model_name" ]; then
        model_name="shuyuej/gemma-2-27b-it-GPTQ"
    fi
    
    echo "Running inference with model: $model_name"
    python "${SCRIPT_DIR}/inference_gemma.py" --input "$input_file" --output "$output_file" --model "$model_name"
}

# Check if input arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 input_file output_file [model_name]"
    exit 1
fi

# Run inference with provided arguments
run_inference "$1" "$2" "$3"