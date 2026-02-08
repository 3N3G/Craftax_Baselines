#!/bin/bash
#SBATCH --job-name=online_rl_hidden
#SBATCH --partition=rl
#SBATCH --gres=gpu:RTX_PRO_6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/online_rl_hidden_%j.out
#SBATCH --error=logs/online_rl_hidden_%j.err

# Online RL with LLM Hidden States
# Uses HuggingFace + Flash Attention for hidden state extraction

set -e

source ~/.bashrc
conda activate craftax_fast_llm

mkdir -p logs
cd ~/Craftax_Baselines

echo "========================================"
echo "Online RL with Hidden States"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --list-gpus | head -1)"
echo "========================================"

# Parameters
ENVS=${1:-8}
STEPS=${2:-2000}
SKIP_N=${3:-1}

echo "Environments: $ENVS"
echo "Steps: $STEPS"
echo "Skip-N: $SKIP_N (LLM inference every $SKIP_N steps)"
echo ""

python online_rl_hidden.py \
    --envs $ENVS \
    --steps $STEPS \
    --skip-n $SKIP_N \
    --wandb-project craftax-online-rl-llm \
    --wandb-entity iris-sobolmark

echo ""
echo "Done!"
