#!/bin/bash
#SBATCH --job-name=benchmark_inference
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# Benchmark LLM inference backends
# Compares: HuggingFace (baseline) vs vLLM vs SGLang

set -e

# Load environment
source ~/.bashrc
conda activate craftax

# Create log directory
mkdir -p logs

# Navigate to project directory  
cd /data/group_data/rl/geney/Craftax_Baselines

# Print system info
echo "========================================"
echo "Inference Benchmark - Job $SLURM_JOB_ID"
echo "========================================"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi --list-gpus)"
echo "Python: $(which python)"
echo "========================================"

# Default: run all backends with 50 samples
# Override with command line args
BACKEND=${1:-huggingface}
SAMPLES=${2:-50}
MAX_TOKENS=${3:-256}

echo "Running benchmark: backend=$BACKEND, samples=$SAMPLES, max_tokens=$MAX_TOKENS"

python benchmark_inference.py \
    --backend $BACKEND \
    --samples $SAMPLES \
    --max-tokens $MAX_TOKENS \
    --batch-size 16 \
    --output logs/benchmark_results_${SLURM_JOB_ID}.json

echo "Benchmark complete! Results saved to logs/benchmark_results_${SLURM_JOB_ID}.json"
