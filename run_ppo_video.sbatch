#!/bin/bash
#SBATCH --job-name=ppo_video
#SBATCH --output=logs/ppo_video_%j.out
#SBATCH --error=logs/ppo_video_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Setup Environment (matching run.sh)
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/craftax

# JAX/Platform Config - USE GPU
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "PPO Training with Video Recording"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

python ppo.py \
    --env_name Craftax-Symbolic-v1 \
    --save_video \
    --save_video_every 4096 \
    --total_timesteps 1e8 \
    --num_envs 128 \
    --wandb_project symbolic_video \
    --wandb_entity iris-sobolmark \
    --video_save_path ./videos

echo "=========================================="
echo "Job Complete: $(date)"
echo "=========================================="
