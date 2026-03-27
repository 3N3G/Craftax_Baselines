#!/bin/bash
#SBATCH --job-name=awr_craftax
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=256G
#SBATCH --open-mode=append

# --- ARGUMENT CHECK ---
# Check if a python script was passed as an argument
if [ -z "$1" ]; then
    echo "Error: You must provide a python script to run."
    echo "Usage: sbatch run_awr.sh <your_script.py>"
    exit 1
fi
PYTHON_SCRIPT=$1
# ----------------------

# 1. Setup Environment
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/test

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_API_KEY="e63622f162f8704ad6a057b040097fa72d663ef6"

# Create a unique temp directory for this specific job to avoid collisions
export WANDB_DIR=$(mktemp -d)
export WANDB_SERVICE_WAIT=300

echo "Starting training on $(hostname) at $(date)"
echo "WandB writing to local dir: $WANDB_DIR"
echo "Running script: $PYTHON_SCRIPT"

# Run the script passed in the argument
python "$PYTHON_SCRIPT"

echo "Finished at $(date)"
