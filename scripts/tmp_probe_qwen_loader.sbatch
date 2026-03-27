#!/bin/bash
#SBATCH --job-name=qwen_loader_probe
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/qwen_loader_probe_%j.out
#SBATCH --error=logs/qwen_loader_probe_%j.err

set -eo pipefail
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/craftax_fast_llm
cd ~/Craftax_Baselines
python3 scripts/tmp_probe_qwen_loader.py
