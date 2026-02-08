#!/bin/bash
#SBATCH --job-name=llm_coordinator
#SBATCH --partition=cpu
#SBATCH --output=/home/geney/Craftax_Baselines/logs/coordinator_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/coordinator_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00                   # Long enough for workers to finish

# This job:
# 1. Starts Redis on THIS compute node
# 2. Writes hostname to shared file for workers
# 3. Queues the labelling jobs
# 4. Waits for all jobs to be processed

set -e

REDIS_PORT=6379
HOSTNAME_FILE="/data/group_data/rl/geney/redis_host.txt"
REDIS_DIR="/home/geney/redis-stable"

echo "=========================================="
echo "LLM Labelling Coordinator"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Setup environment
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/imaug

# 1. Start Redis on this node (disabled protected mode for cluster access)
echo "Starting Redis on $(hostname):${REDIS_PORT}..."
cd $REDIS_DIR
./src/redis-server --port $REDIS_PORT --bind 0.0.0.0 --protected-mode no --daemonize yes
sleep 2

# Verify Redis is running
./src/redis-cli -p $REDIS_PORT ping || { echo "Redis failed to start!"; exit 1; }
echo "Redis running!"

# 2. Write hostname for workers
echo "$(hostname)" > $HOSTNAME_FILE
echo "Wrote hostname to $HOSTNAME_FILE"

# DEBUG: Test Redis connection multiple ways
echo "=== DEBUG: Testing Redis connectivity ==="
echo "Hostname: $(hostname)"
echo "Testing localhost..."
./src/redis-cli -h localhost -p $REDIS_PORT ping
echo "Testing 127.0.0.1..."
./src/redis-cli -h 127.0.0.1 -p $REDIS_PORT ping
echo "Testing $(hostname)..."
./src/redis-cli -h $(hostname) -p $REDIS_PORT ping

# DEBUG: Test Python Redis connection
python -c "
import redis
import socket
hostname = socket.gethostname()
print(f'Testing Python redis connection to {hostname}:6379...')
r = redis.Redis(host=hostname, port=6379, decode_responses=True)
r.ping()
print('Python redis connection: SUCCESS')
r.lpush('test_queue', 'test_value')
print('Python lpush: SUCCESS')
r.delete('test_queue')
print('Python delete: SUCCESS')
"

# 3. Clear any stale queue entries first
echo "Clearing any stale queue entries..."
redis-cli -h $(hostname) -p $REDIS_PORT DEL craftax_llm_job_queue || true

# 4. Queue all the jobs
echo "Queuing labelling jobs..."
cd /home/geney/Craftax_Baselines/labelling
python addtoqueue_llm.py --host $(hostname) --symbolic --queue craftax_llm_job_queue

# 5. Check initial queue size
INITIAL_SIZE=$(redis-cli -h $(hostname) -p $REDIS_PORT LLEN craftax_llm_job_queue)
echo "Queued $INITIAL_SIZE jobs"

if [ "$INITIAL_SIZE" -eq 0 ]; then
    echo "No jobs to process! All files already completed."
    redis-cli -h $(hostname) -p $REDIS_PORT SHUTDOWN NOSAVE || true
    exit 0
fi

# 6. Monitor loop - stay alive until queue is empty AND workers have finished
echo "Monitoring queue... Workers should be started separately with makeworkers_llm.sbatch"
echo "Queue will be checked every 60 seconds"

ZERO_COUNT=0
ZERO_WAIT_CYCLES=5  # Wait 5 more minutes after queue empties for in-flight jobs

while true; do
    REMAINING=$(redis-cli -h $(hostname) -p $REDIS_PORT LLEN craftax_llm_job_queue)
    
    # Check if any vllm_worker jobs are still running
    ACTIVE_WORKERS=$(squeue -u $USER -n vllm_labeller -h 2>/dev/null | wc -l)
    
    echo "$(date): $REMAINING jobs in queue, $ACTIVE_WORKERS active workers"
    
    if [ "$REMAINING" -eq 0 ]; then
        ZERO_COUNT=$((ZERO_COUNT + 1))
        echo "  Queue empty for $ZERO_COUNT cycles..."
        
        # Only exit if queue is empty AND no workers are running
        if [ "$ACTIVE_WORKERS" -eq 0 ]; then
            echo "All jobs completed and all workers finished!"
            break
        fi
        
        # Safety: if queue empty for too long but workers exist, they might be stuck
        if [ "$ZERO_COUNT" -ge "$ZERO_WAIT_CYCLES" ] && [ "$ACTIVE_WORKERS" -gt 0 ]; then
            echo "Warning: Queue empty but $ACTIVE_WORKERS workers still running. Waiting for them..."
        fi
    else
        ZERO_COUNT=0
    fi
    
    sleep 60
done

# Cleanup
echo "Shutting down Redis..."
redis-cli -h $(hostname) -p $REDIS_PORT SHUTDOWN NOSAVE || true

echo "=========================================="
echo "Coordinator Complete: $(date)"
echo "=========================================="
