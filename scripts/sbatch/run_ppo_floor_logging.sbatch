#!/bin/bash
#SBATCH --job-name=ppo_floor_log
#SBATCH --output=logs/ppo_floor_log_%j.out
#SBATCH --error=logs/ppo_floor_log_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Setup Environment
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/craftax

# JAX/Platform Config - USE GPU
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "PPO Symbolic with Floor Logging"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

python online_rl/ppo_floor_logging.py \
    --env_name Craftax-Symbolic-v1 \
    --total_timesteps 1e8 \
    --num_envs 128 \
    --wandb_project symbolic_labelling \
    --wandb_entity iris-sobolmark \
    --save_policy

echo "=========================================="
echo "Job Complete: $(date)"
echo "=========================================="
