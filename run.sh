#!/bin/bash
#SBATCH --job-name=generic_run
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Setup Environment
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/test

# JAX/Platform Config (Keeping JAX off GPU to save memory for PyTorch)
export JAX_PLATFORMS="cpu"
export JAX_PLATFORM_NAME="cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1

# W&B Config
export WANDB_API_KEY="e63622f162f8704ad6a057b040097fa72d663ef6"
export WANDB_DIR=$(mktemp -d)

echo "=========================================="
echo "Generic Batch Runner"
echo "=========================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Running script: $1"
echo "=========================================="
echo ""

# Run the python script passed as arguments
# "$@" passes all arguments provided to sbatch (e.g. script.py --arg 1)
python "$@"

echo ""
echo "=========================================="
echo "Job Complete"
echo "Finished at: $(date)"
echo "=========================================="
