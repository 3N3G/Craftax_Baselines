#!/bin/bash
#SBATCH --job-name=hidden_bench
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/hidden_bench_%j.out
#SBATCH --error=logs/hidden_bench_%j.err

# Benchmark hidden state extraction methods
# Compares HuggingFace, vLLM+HF hybrid, etc.

set -e

source ~/.bashrc
conda activate craftax_fast_llm

mkdir -p logs
cd ~/Craftax_Baselines

echo "========================================"
echo "Hidden State Benchmark - Job $SLURM_JOB_ID"
echo "========================================"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --list-gpus | head -1)"
echo "========================================"

SAMPLES=${1:-16}
BATCH_SIZE=${2:-8}
BACKEND=${3:-all}

echo "Samples: $SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Backend: $BACKEND"
echo ""

python benchmarks/benchmark_hidden_states.py \
    --backend $BACKEND \
    --samples $SAMPLES \
    --batch-size $BATCH_SIZE

echo ""
echo "Done! Results saved to hidden_state_benchmark_*.json"
