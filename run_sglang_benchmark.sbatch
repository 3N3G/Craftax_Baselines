#!/bin/bash
#SBATCH --job-name=sglang_bench
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/sglang_bench_%j.out
#SBATCH --error=logs/sglang_bench_%j.err

# SGLang Benchmark - Tests RadixAttention advantage for RL workloads
# This properly benchmarks SGLang with batched async requests

set -e

source ~/.bashrc
conda activate craftax_fast_llm

mkdir -p logs
cd ~/Craftax_Baselines

echo "========================================"
echo "SGLang Benchmark - Job $SLURM_JOB_ID"
echo "========================================"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --list-gpus | head -1)"
echo "========================================"

PORT=30000
SAMPLES=${1:-50}

# Start SGLang server
echo "Starting SGLang server..."
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Thinking-2507 \
    --port 30000 \
    --context-length 4096 \
    --dtype float16 \
    --trust-remote-code &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server
echo "Waiting for server..."
for i in {1..90}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}0 seconds"
        break
    fi
    sleep 10
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "ERROR: Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Run benchmark
echo ""
echo "Running SGLang benchmark with $SAMPLES samples..."
python benchmark_inference.py \
    --backend sglang \
    --samples $SAMPLES \
    --max-tokens 256 \
    --server-url http://localhost:$PORT \
    --output logs/sglang_results_${SLURM_JOB_ID}.json

# Also run a quick batched test using the smoke test
echo ""
echo "Running batched RL simulation (4 envs, 25 steps)..."
JAX_PLATFORM_NAME=cpu python online_rl_smoke_test.py \
    --envs 4 \
    --steps 25 \
    --server-url http://localhost:$PORT

# Cleanup
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null

echo "Done! Results in logs/sglang_results_${SLURM_JOB_ID}.json"
