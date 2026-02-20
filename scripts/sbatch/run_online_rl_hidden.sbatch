#!/bin/bash
#SBATCH --job-name=online_rl_hidden
#SBATCH --partition=rl
#SBATCH --gres=gpu:RTX_PRO_6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/online_rl_hidden_%j.out
#SBATCH --error=logs/online_rl_hidden_%j.err

# Online RL with vLLM Hidden States (34x faster than HuggingFace)

set -e

source ~/.bashrc
conda activate craftax_fast_llm  # Updated to use vLLM environment

mkdir -p logs
cd ~/Craftax_Baselines

echo "========================================"
echo "Online RL with vLLM Hidden States"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --list-gpus | head -1)"
echo "========================================"

# Parameters
ENVS=${1:-8}
TIMESTEPS=${2:-1000000}
SKIP_N=${3:-1}
LAYER=${4:--1}  # Which layer to extract (-1 for last available, or 0-35 for specific layer)
TOKENS=${5:-1}  # How many tokens to generate (1 for prompt-only, more for generation)

echo "Environments: $ENVS"
echo "Timesteps: $TIMESTEPS"
echo "Skip-N: $SKIP_N (LLM inference every $SKIP_N steps)"
echo "Extract layer: $LAYER (-1 means last of extracted layers)"
echo "Tokens to generate: $TOKENS"
echo ""

# Start vLLM server in background
echo "Starting vLLM server..."
bash scripts/start_vllm_hidden.sh --mode last_token > logs/vllm_server_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

# Wait for server to be ready (up to 3 minutes)
echo "Waiting for vLLM server to initialize (this takes ~2 minutes)..."
for i in {1..180}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready!"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: vLLM server failed to start after 3 minutes"
        cat logs/vllm_server_${SLURM_JOB_ID}.log
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Still waiting... ($i seconds)"
    fi
done

# Run training
python online_rl_llm/online_rl_hidden.py \
    --envs $ENVS \
    --timesteps $TIMESTEPS \
    --skip-n $SKIP_N \
    --layer $LAYER \
    --tokens $TOKENS \
    --wandb-project craftax-online-rl-llm \
    --wandb-entity iris-sobolmark

# Clean up - kill vLLM server
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true

echo ""
echo "Done!"
