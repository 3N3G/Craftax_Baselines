#!/bin/bash
#SBATCH --job-name=vlm_server
#SBATCH --output=logs/vlm_server_%j.out
#SBATCH --error=logs/vlm_server_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# VLM Server for Augmented AWR Evaluation
# This hosts Qwen3-VL-4B model and provides API for hidden state generation

# Setup Environment
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/test

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Starting VLM Server"
echo "=========================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Port: 5000"
echo "=========================================="
echo ""

# Get hostname for client reference
HOSTNAME=$(hostname)
echo "VLM Server will be available at: http://${HOSTNAME}:5000"
echo "Pass this hostname to eval_batch_augmented.sh"
echo ""

# Start server
python vlm_server.py --host 0.0.0.0 --port 5000
