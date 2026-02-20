#!/bin/bash
#SBATCH --job-name=vllm_rl_test
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/vllm_rl_%j.out
#SBATCH --error=logs/vllm_rl_%j.err

# Online RL Test with vLLM
# Tests throughput and estimates training time

set -e

source ~/.bashrc
conda activate craftax_fast_llm

mkdir -p logs
cd ~/Craftax_Baselines

echo "========================================"
echo "vLLM Online RL Test - Job $SLURM_JOB_ID"
echo "========================================"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --list-gpus | head -1)"
echo "========================================"

ENVS=${1:-8}
STEPS=${2:-50}

echo ""
echo "Running with $ENVS envs, $STEPS steps..."
echo ""

JAX_PLATFORM_NAME=cpu python online_rl_llm/smoke_test_vllm.py \
    --envs $ENVS \
    --steps $STEPS

echo ""
echo "Done!"
