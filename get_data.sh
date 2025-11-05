#!/bin/bash
#SBATCH --job-name=craftax_ppo
#SBATCH --output=logs/craftax_%j.out
#SBATCH --error=logs/craftax_%j.err
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=48:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate conda environment
source /home/geney/anaconda3/bin/activate
conda activate craftax

# Navigate to working directory
cd ~/Craftax_Baselines

# Run the training
python ppo.py \
    --save_traj

echo "Job finished at: $(date)"