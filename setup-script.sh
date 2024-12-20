#!/bin/bash

echo "Setting up the environment..."

# Install base requirements
pip install -U "optimum>=1.20.0"
pip install accelerate 
pip install datasets 
pip install jsonlines 
pip install sentencepiece

# Install PyTorch and transformers
pip install torch
pip install transformers
pip install auto-gptq
pip install huggingface_hub
pip install openai

echo "Environment setup completed."
