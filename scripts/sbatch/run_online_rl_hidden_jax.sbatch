#!/bin/bash
#SBATCH --job-name=online_rl_hidden_jax
#SBATCH --partition=rl
#SBATCH --gres=gpu:RTX_PRO_6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/online_rl_hidden_jax_%j.out
#SBATCH --error=logs/online_rl_hidden_jax_%j.err

# Online RL with vLLM Hidden States - JAX implementation.

set -eo pipefail

# Source bashrc defensively because some profile scripts can return non-zero.
set +e
set +u
set +o pipefail
source ~/.bashrc
BASHRC_RC=$?
set -e
set -u
set -o pipefail
if [[ $BASHRC_RC -ne 0 ]]; then
    echo "WARNING: source ~/.bashrc exited with code $BASHRC_RC; continuing"
fi

mkdir -p logs
cd ~/Craftax_Baselines
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found after sourcing bashrc"
    exit 1
fi

activate_conda_for() {
    local purpose="$1"
    shift
    local env_name
    for env_name in "$@"; do
        if conda activate "${env_name}" >/dev/null 2>&1; then
            echo "Activated conda env for ${purpose}: ${env_name}"
            return 0
        fi
    done
    echo "ERROR: failed to activate conda env for ${purpose}"
    echo "Tried: $*"
    conda env list || true
    return 1
}

# vLLM server needs craftax_fast_llm, training needs craftax (for distrax)
conda activate craftax_fast_llm

# Parameters
ENVS=${1:-8}
TIMESTEPS=${2:-1000000}
SKIP_N=${3:-1}
LAYER=${4:--1}  # Which layer to extract (-1 for last available)
TOKENS=${5:-1}  # How many tokens to generate (1 for prompt-only)
NUM_STEPS=${6:-64}

echo "========================================"
echo "Online RL with vLLM Hidden States (JAX)"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --list-gpus | head -1)"
echo "========================================"

echo "Environments: $ENVS"
echo "Timesteps: $TIMESTEPS"
echo "Skip-N: $SKIP_N (LLM inference every $SKIP_N steps)"
echo "Num-steps/update: $NUM_STEPS"
echo "Extract layer: $LAYER"
echo "Tokens to generate: $TOKENS"
echo ""

VLLM_PID=""
cleanup() {
    if [[ -n "${VLLM_PID}" ]]; then
        echo "Stopping vLLM server (pid=${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Start vLLM server in background
echo "Starting vLLM server..."
activate_conda_for "vllm-server" \
    /data/user_data/geney/.conda/envs/craftax_fast_llm \
    craftax_fast_llm \
    base
bash scripts/start_vllm_hidden.sh --mode last_token > logs/vllm_server_jax_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

# Wait for server to be ready (no timeout - will wait as long as needed)
echo "Waiting for vLLM server to initialize (this can take several minutes)..."
echo "The server needs to load all model shards before starting..."
SECONDS=0
while true; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ vLLM server ready after $SECONDS seconds!"
        break
    fi

    # Check if the vLLM process is still running
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ ERROR: vLLM server process died unexpectedly"
        echo "Showing last 50 lines of server log:"
        tail -50 logs/vllm_server_jax_${SLURM_JOB_ID}.log
        exit 1
    fi

    sleep 2
    if [ $((SECONDS % 30)) -eq 0 ] && [ $SECONDS -gt 0 ]; then
        echo "  Still waiting... ($SECONDS seconds elapsed)"
        # Show a snippet of the log to see progress
        echo "  Latest from server log:"
        tail -3 logs/vllm_server_jax_${SLURM_JOB_ID}.log | sed 's/^/    /'
    fi
done

# Switch to craftax environment for JAX training (has distrax)
echo ""
echo "Switching to craftax environment for training..."
activate_conda_for "training" \
    /data/user_data/geney/.conda/envs/craftax \
    craftax \
    base

# Run JAX version
echo "Starting JAX training..."
echo "========================================"
python -u online_rl_llm/online_rl_hidden_jax.py \
    --envs $ENVS \
    --timesteps $TIMESTEPS \
    --skip-n $SKIP_N \
    --num-steps $NUM_STEPS \
    --layer $LAYER \
    --tokens $TOKENS \
    --wandb-project craftax-online-rl-llm \
    --wandb-entity iris-sobolmark

echo ""
echo "Done! Date: $(date)"
