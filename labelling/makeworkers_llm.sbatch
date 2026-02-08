#!/bin/bash

#SBATCH --job-name=llm_worker
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos             
#SBATCH --array=1-30                   # Default, can be overridden with --array=1-N
#SBATCH --output=/home/geney/Craftax_Baselines/logs/llm_worker_%A_%a.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/llm_worker_%A_%a.err
#
# --- Resource Request ---
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1              # Text-only LLM, may need less memory than VLM
#
# --- Optional ---
#SBATCH --time=24:00:00
#SBATCH --mem=256G


echo "Starting LLM worker on node $(hostname)"

source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/imaug
cd /home/geney/Craftax_Baselines/labelling

srun --cpu-bind=none python llm_worker.py
