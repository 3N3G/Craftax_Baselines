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
#SBATCH --gres=gpu:L40S:1
#
# --- Optional ---
#SBATCH --time=24:00:00
#SBATCH --mem=64G

echo "Starting LLM worker on node $(hostname)"

source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/craftax_fast_llm

cd /home/geney/Craftax_Baselines

# Start vLLM server on this node (each worker has its own GPU + server)
echo "Starting vLLM server..."
bash scripts/start_vllm_hidden.sh --mode last_token > logs/vllm_worker_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
VLLM_PID=$!

# Wait for vLLM server to be ready (up to 3 minutes)
echo "Waiting for vLLM server..."
for i in {1..180}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready!"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: vLLM server failed to start after 3 minutes"
        cat logs/vllm_worker_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Run the worker
cd /home/geney/Craftax_Baselines/labelling
srun --cpu-bind=none python llm_worker.py

# Cleanup
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
echo "Worker done."

