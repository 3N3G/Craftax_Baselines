#!/bin/bash
#SBATCH --job-name=eval_wave_id
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/geney/Craftax_Baselines/logs/eval_wave_id_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/eval_wave_id_%j.err

set -euo pipefail
export TRACKS="${TRACKS:-id}"
export START_VLLM="${START_VLLM:-1}"
exec bash scripts/sbatch/run_eval_policy_wave.sbatch
