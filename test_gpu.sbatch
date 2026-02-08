#!/bin/bash
#SBATCH --job-name=ppo_test_gpu
#SBATCH --output=logs/ppo_test_gpu_%j.out
#SBATCH --error=logs/ppo_test_gpu_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Setup Environment
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/craftax # IMPORTANT FOR JAX VERSIONING

# JAX/Platform Config - USE GPU (no CPU forcing)
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "PPO GPU Test Run"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Short test run with different output directory
python ppo.py \
    --env_name Craftax-Symbolic-v1 \
    --save_traj \
    --traj_save_path /data/group_data/rl/geney/craftax_test_gpu/ \
    --total_timesteps 1e6 \
    --num_envs 128 \
    --save_traj_every 1 \
    --no-use_wandb

echo "=========================================="
echo "Test Complete: $(date)"
echo "=========================================="
