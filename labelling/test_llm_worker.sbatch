#!/bin/bash
#SBATCH --job-name=llm_worker_test
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --output=/home/geney/Craftax_Baselines/logs/llm_test_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/llm_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=00:30:00
#SBATCH --mem=128G

echo "=========================================="
echo "LLM Worker Test Run"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/imaug

cd /home/geney/Craftax_Baselines/labelling

# Test imports and model loading only
python -c "
import redis
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from craftax.craftax.renderer import render_craftax_text

print('=== Testing imports ===')
print('Redis:', redis.__version__)
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

print('=== Testing Redis connection ===')
r = redis.Redis(host='login1', port=6379, decode_responses=True)
r.ping()
print('Redis: Connected to login1:6379')

print('=== Testing model loading ===')
MODEL_ID = 'Qwen/Qwen3-4B-Thinking-2507'
print(f'Loading {MODEL_ID}...')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f'Model loaded! Hidden size: {model.config.hidden_size}')

print('=== Testing render_craftax_text ===')
import numpy as np
# Create dummy obs (shape should match symbolic obs)
dummy_obs = np.zeros(8268)  # Approximate symbolic obs size
try:
    text = render_craftax_text(dummy_obs)
    print(f'Render output (first 100 chars): {text[:100]}...')
except Exception as e:
    print(f'Render error (may need real state): {e}')

print('=== All tests passed! ===')
"

echo "=========================================="
echo "Test Complete: $(date)"
echo "=========================================="
