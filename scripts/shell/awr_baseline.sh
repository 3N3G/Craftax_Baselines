#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=logs/awr_baseline_%j.log
#SBATCH --error=logs/awr_baseline_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate test

echo "Starting baseline AWR training to 100k steps..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python offline_rl/awr.py \
    --save_dir "/data/group_data/rl/geney/checkpoints/awr_baseline_v2/" \
    --wandb_name "awr-baseline-100k-v2" \
    --eval_freq 1000000

echo "End time: $(date)"
echo "Training complete!"
