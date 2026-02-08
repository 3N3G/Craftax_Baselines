#!/bin/bash
#SBATCH --job-name=ppo_symbolic
#SBATCH --output=logs/ppo_symbolic_%j.out
#SBATCH --error=logs/ppo_symbolic_%j.err
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
echo "PPO Symbolic Data Generation"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

python ppo.py \
    --env_name Craftax-Symbolic-v1 \
    --save_traj \
    --total_timesteps 1e8 \
    --num_envs 128 \
    --save_traj_every 10 \
    --wandb_project symbolic_labelling \
    --wandb_entity iris-sobolmark

echo "=========================================="
echo "Job Complete: $(date)"
echo "=========================================="
