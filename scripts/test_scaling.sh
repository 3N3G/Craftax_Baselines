#!/bin/bash
# Test scaling online RL to 128 environments

echo "Testing online RL scaling with fixed text rendering"
echo "===================================================="

# Test different configurations
for ENVS in 8 16 32 64 128; do
    echo ""
    echo "Testing with $ENVS environments (skip_n=25)..."
    python online_rl_llm/online_rl_hidden.py \
        --envs $ENVS \
        --steps 100 \
        --skip-n 25 \
        --no-wandb \
        --quiet 2>&1 | grep -E "Samples/sec|LLM inferences"
done

echo ""
echo "Memory usage should be fine:"
echo "- JAX state: ~11 MB for 128 envs"
echo "- PyTorch policy: ~7 MB"
echo "- vLLM concurrent requests: ~6.4 GB (well within 32.7 GB available)"
echo ""
echo "Expected performance:"
echo "- With skip_n=25, text rendering happens 4% of the time"
echo "- Should achieve 100+ SPS with 128 environments"