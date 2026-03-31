#!/bin/bash
#SBATCH --job-name=clone_qwen35_env
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/clone_qwen35_env_%j.out
#SBATCH --error=logs/clone_qwen35_env_%j.err

set -eo pipefail
cd ~/Craftax_Baselines

bash scripts/shell/clone_qwen35_env.sh
