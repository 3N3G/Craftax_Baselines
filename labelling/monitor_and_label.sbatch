#!/bin/bash
#SBATCH --job-name=monitor_and_label
#SBATCH --partition=cpu
#SBATCH --output=/home/geney/Craftax_Baselines/logs/monitor_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/monitor_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=72:00:00

# This script monitors a directory for data collection completion,
# then starts the coordinator and workers for LLM labelling.

DATA_DIR="${1:-/data/group_data/rl/geney/craftax_unlabelled_symbolic_with_text/}"
TARGET_COUNT="${2:-1221}"
WAIT_INTERVAL=60  # Check every 60 seconds

echo "=========================================="
echo "Data Collection Monitor + Auto-Labelling"
echo "Monitoring: $DATA_DIR"
echo "Target file count: $TARGET_COUNT"
echo "Date: $(date)"
echo "=========================================="

# Wait for data collection to complete
while true; do
    if [ -d "$DATA_DIR" ]; then
        CURRENT_COUNT=$(ls -1 "$DATA_DIR"/*.npz 2>/dev/null | wc -l)
    else
        CURRENT_COUNT=0
    fi
    
    echo "$(date): $CURRENT_COUNT / $TARGET_COUNT files"
    
    if [ "$CURRENT_COUNT" -ge "$TARGET_COUNT" ]; then
        echo "Target reached! Starting labelling pipeline..."
        break
    fi
    
    sleep $WAIT_INTERVAL
done

# Give extra time for last file to finish writing
echo "Waiting 30s for files to finalize..."
sleep 30

# Start coordinator
echo "Starting coordinator..."
cd /home/geney/Craftax_Baselines/labelling
COORD_JOB=$(sbatch --parsable coordinator.sbatch)
echo "Coordinator job ID: $COORD_JOB"

# Wait for coordinator to start Redis and queue jobs
echo "Waiting 60s for coordinator to initialize..."
sleep 60

# Start workers
echo "Starting workers..."
WORKER_JOB=$(sbatch --parsable makeworkers_llm.sbatch)
echo "Worker job ID: $WORKER_JOB"

echo "=========================================="
echo "Labelling pipeline started!"
echo "Coordinator: $COORD_JOB"
echo "Workers: $WORKER_JOB"
echo "=========================================="

# Monitor until queue is empty (optional - coordinator already does this)
echo "Monitoring queue progress..."
while true; do
    REDIS_HOST=$(cat /data/group_data/rl/geney/redis_host.txt 2>/dev/null)
    if [ -n "$REDIS_HOST" ]; then
        REMAINING=$(redis-cli -h "$REDIS_HOST" LLEN craftax_llm_job_queue 2>/dev/null || echo "?")
        echo "$(date): $REMAINING jobs remaining"
        
        if [ "$REMAINING" = "0" ]; then
            echo "All jobs completed!"
            break
        fi
    fi
    sleep 120
done

echo "=========================================="
echo "Pipeline Complete: $(date)"
echo "=========================================="
