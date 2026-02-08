#!/bin/bash

#SBATCH --job-name=vllm_labelling
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos             
#SBATCH --array=1-32                   # Submit 32 tasks
#SBATCH --output=/home/geney/Craftax_Baselines/logs/vllm_worker_%A_%a.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/vllm_worker_%A_%a.err
#
# --- Resource Request ---
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#
# --- Optional ---
#SBATCH --time=24:00:00
#SBATCH --mem=256G


echo "Starting vLLM worker on node $(hostname)"

source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/craftax_fast_llm

cd /home/geney/Craftax_Baselines

srun --cpu-bind=none python labelling/vllm_labeller.py --queue-mode
