#!/bin/bash
#SBATCH --job-name=online_rl_test
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/online_rl_%j.out
#SBATCH --error=logs/online_rl_%j.err

# Online RL Smoke Test with SGLang
# Runs SGLang server on GPU 0, Craftax RL loop on GPU 1

set -e

# Load environment
source ~/.bashrc
conda activate craftax

# Create log directory
mkdir -p logs

cd /data/group_data/rl/geney/Craftax_Baselines

echo "========================================"
echo "Online RL Smoke Test - Job $SLURM_JOB_ID"
echo "========================================"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi --list-gpus)"
echo "========================================"

# Parameters
ENVS=${1:-16}
STEPS=${2:-100}
PORT=30000

# Start SGLang server on GPU 0 in background
echo "Starting SGLang server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-4B-Thinking-2507 \
    --port $PORT \
    --context-length 4096 \
    --enable-radix-caching \
    --dtype float16 \
    --trust-remote-code &

SERVER_PID=$!
echo "SGLang server started (PID: $SERVER_PID)"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    echo "  Waiting... ($i/60)"
    sleep 5
done

# Verify server is up
if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "ERROR: Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Run smoke test on GPU 1 (JAX uses CPU for Craftax in this setup)
echo ""
echo "Running online RL smoke test..."
echo "  Environments: $ENVS"
echo "  Steps: $STEPS"
echo ""

JAX_PLATFORM_NAME=cpu python online_rl_llm/smoke_test_sglang.py \
    --envs $ENVS \
    --steps $STEPS \
    --server-url http://localhost:$PORT

# Cleanup
echo ""
echo "Stopping SGLang server..."
kill $SERVER_PID 2>/dev/null

echo "Done!"
