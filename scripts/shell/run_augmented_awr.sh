#!/bin/bash
#SBATCH --job-name=augmented_awr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=256G
#SBATCH --open-mode=append

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

echo "=============================================="
echo "FULL RUN: Augmented AWR (awr_aug.py)"
echo "=============================================="
echo "Starting on $(hostname) at $(date)"
echo "WandB writing to local dir: $WANDB_DIR"
echo ""

python offline_rl/awr_augmented.py \
  --data_dir /data/group_data/rl/geney/craftax_labelled_results_with_returns \
  --save_dir /data/group_data/rl/geney/checkpoints/awr_augmented_v2/ \
  --total_steps 100000 \
  --batch_size 256 \
  --lr 3e-4 \
  --awr_beta 10.0 \
  --seed 42 \
  --wandb_name awr-augmented-onlygeneration \
  --save_freq 25000

echo ""
echo "=============================================="
echo "TRAINING COMPLETED"
echo "Finished at $(date)"
echo "=============================================="
