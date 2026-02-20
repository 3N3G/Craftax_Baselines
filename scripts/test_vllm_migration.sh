#!/bin/bash
# Test script for vLLM hidden states migration
# Run this on a GPU node (babel)

set -e

echo "================================"
echo "vLLM Hidden States Migration Test"
echo "================================"

# 1. Check environment
echo -e "\n1. Checking environment..."
conda info | grep "active environment"
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# 2. Test current setup with Qwen3-4B-Thinking
echo -e "\n2. Testing with current model (Qwen3-4B-Thinking-2507)..."
echo "Starting vLLM server..."

# Kill any existing server
pkill -f "vllm serve" || true
sleep 2

# Start server in background
export STORAGE_PATH="/tmp/hidden_states_test"
rm -rf $STORAGE_PATH
mkdir -p $STORAGE_PATH

echo "Launching vLLM server..."
nohup vllm serve "./configs/vllm_hidden_qwen4b" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"'$STORAGE_PATH'","mode":"last_token"}}' \
    > vllm_test.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to initialize (this takes ~2 minutes)..."
for i in {1..120}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Server failed to start after 2 minutes"
        cat vllm_test.log
        kill $SERVER_PID || true
        exit 1
    fi
    sleep 1
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Still waiting... ($i seconds)"
    fi
done

# 3. Test extraction
echo -e "\n3. Testing hidden state extraction..."
python scripts/test_vllm_hidden.py

# 4. Compare with HuggingFace
echo -e "\n4. Comparing extraction methods..."
python scripts/benchmark_hidden_no_cot.py --backend both --num-prompts 10

# 5. Test with utils.llm_extractor
echo -e "\n5. Testing VLLMHiddenStateExtractor class..."
python -c "
import sys
sys.path.insert(0, '.')
from utils.llm_extractor import VLLMHiddenStateExtractor
import numpy as np

print('Creating extractor...')
extractor = VLLMHiddenStateExtractor(
    server_url='http://localhost:8000',
    model_name='./configs/vllm_hidden_qwen4b',
    target_layer=-1  # Last layer
)

print('Testing extraction...')
observations = ['The player sees a tree at position 1,0', 'Health is low at 3.0']
hidden_states, metrics = extractor.extract_hidden_states_no_cot(observations)

print(f'Hidden states shape: {hidden_states.shape}')
print(f'Hidden states stats: mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}')
print(f'Metrics: {metrics}')
"

# Clean up
echo -e "\n6. Cleaning up..."
kill $SERVER_PID || true
rm -rf $STORAGE_PATH

echo -e "\n================================"
echo "Test complete!"
echo "Check vllm_test.log for server output"
echo "================================"