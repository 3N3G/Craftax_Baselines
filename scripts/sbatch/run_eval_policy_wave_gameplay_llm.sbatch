#!/bin/bash
#SBATCH --job-name=eval_wave_llm
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/geney/Craftax_Baselines/logs/eval_wave_llm_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/eval_wave_llm_%j.err

set -euo pipefail
export TRACKS="${TRACKS:-gameplay_llm}"
export START_VLLM="${START_VLLM:-1}"
export LLM_GENERATION_TOKENS="${LLM_GENERATION_TOKENS:-32}"
exec bash scripts/sbatch/run_eval_policy_wave.sbatch
