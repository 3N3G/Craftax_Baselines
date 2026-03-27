#!/bin/bash
#SBATCH --job-name=eval_aug_batch
#SBATCH --output=logs/eval_aug_batch_%j.out
#SBATCH --error=logs/eval_aug_batch_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

SERVER_HOST="${1:-babel-t9-20}"
SERVER_PORT=5000

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
echo "AWR AUGMENTED Batch Evaluation"
echo "=========================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Server URL: http://${SERVER_HOST}:${SERVER_PORT}"
echo "Evaluating: 3 episodes per checkpoint"
echo "Visualization: Dual line graph (Value + RTG)"
echo "=========================================="
echo ""

CHECKPOINT_DIR="/data/group_data/rl/geney/checkpoints/awr_augmented"
STATS_FILE="${CHECKPOINT_DIR}/hidden_state_stats.npz"

# Check if stats file exists
if [ ! -f "$STATS_FILE" ]; then
    echo "ERROR: Normalization stats file not found: $STATS_FILE"
    echo "Please run: python tools/compute_hidden_stats.py --data_dir /data/group_data/rl/geney/craftax_labelled_results_with_returns --output $STATS_FILE"
    exit 1
fi

echo "✓ Found normalization stats: $STATS_FILE"
echo ""

CHECKPOINTS=(
#    "awr_aug_checkpoint_25000.pth"
#    "awr_aug_checkpoint_50000.pth"
#    "awr_aug_checkpoint_75000.pth"
    "awr_aug_checkpoint_100000.pth"
)

for checkpoint in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating: $checkpoint"
    echo "Time: $(date)"
    echo "=========================================="

    python offline_rl/eval_awr_augmented.py \
        --checkpoint "${CHECKPOINT_DIR}/${checkpoint}" \
        --server_url "http://${SERVER_HOST}:${SERVER_PORT}" \
        --num_episodes 3 \
        --save_video \
        --video_dir "./eval_videos_proper/${checkpoint%.pth}" \
        --stats_path "$STATS_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully evaluated $checkpoint"
    else
        echo "✗ Failed to evaluate $checkpoint"
    fi
done

echo ""
echo "=========================================="
echo "All Augmented Evaluations Complete!"
echo "Finished at: $(date)"
echo "=========================================="
