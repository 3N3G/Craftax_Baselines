#!/bin/bash
#SBATCH --job-name=eval_baseline_batch
#SBATCH --output=logs/eval_baseline_batch_%j.out
#SBATCH --error=logs/eval_baseline_batch_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Setup Environment
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/test

export JAX_PLATFORMS="cpu"
export JAX_PLATFORM_NAME="cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1

export WANDB_API_KEY="e63622f162f8704ad6a057b040097fa72d663ef6"
export WANDB_DIR=$(mktemp -d)

echo "=========================================="
echo "AWR BASELINE Batch Evaluation"
echo "=========================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Evaluating: 3 episodes per checkpoint"
echo "Visualization: Value bar (simple)"
echo "=========================================="
echo ""

CHECKPOINT_DIR="/data/group_data/rl/geney/checkpoints/awr_baseline_v2"

CHECKPOINTS=(
#    "awr_checkpoint_25000.pth"
#    "awr_checkpoint_50000.pth"
#    "awr_checkpoint_75000.pth"
     "awr_checkpoint_100000.pth"
)

for checkpoint in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating: $checkpoint"
    echo "Time: $(date)"
    echo "=========================================="

    python offline_rl/eval_awr.py \
        --checkpoint "${CHECKPOINT_DIR}/${checkpoint}" \
        --env_name "Craftax-Pixels-v1" \
        --num_episodes 100 \
        --wandb_project "craftax-offline-awr" \
        --wandb_entity "iris-sobolmark" \

    if [ $? -eq 0 ]; then
        echo "✓ Successfully evaluated $checkpoint"
    else
        echo "✗ Failed to evaluate $checkpoint"
    fi
done

echo ""
echo "=========================================="
echo "All Baseline Evaluations Complete!"
echo "Finished at: $(date)"
echo "=========================================="
