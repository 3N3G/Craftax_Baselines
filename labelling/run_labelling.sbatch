#!/bin/bash
#SBATCH --job-name=llm_label_all
#SBATCH --partition=cpu
#SBATCH --output=/home/geney/Craftax_Baselines/logs/label_all_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/label_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

###############################################################################
# UNIFIED OFFLINE LABELLING SCRIPT
#
# This script handles EVERYTHING:
# 1. Starts Redis coordinator
# 2. Scans for pending files (skips already completed)
# 3. Submits GPU workers that extract HIDDEN STATES (not vLLM!)
# 4. Monitors until all jobs complete
# 5. Cleans up
#
# CRITICAL: Uses llm_worker.py which saves HIDDEN STATES (a necessity for offline training augmented policies)
#
# Usage:
#   sbatch labelling/run_labelling.sbatch
#
# To check progress:
#   ls /data/group_data/rl/geney/craftax_llm_labelled_results/*.npz | wc -l
#
# Optional environment variables (set before sbatch or in script):
#   RESULTS_DIR - Where to save results (default: /data/group_data/rl/geney/craftax_llm_labelled_results)
#   SOURCE_DIR  - Where to find input files (default: /data/group_data/rl/geney/craftax_unlabelled_symbolic)
#   NUM_WORKERS - Number of GPU workers to spawn (default: 16)
###############################################################################

set -e

# =============================================================================
# CONFIGURATION - Edit these or override via environment variables
# =============================================================================
RESULTS_DIR="${RESULTS_DIR:-/data/group_data/rl/geney/new_craftax_llm_labelled_results}"
SOURCE_DIR="${SOURCE_DIR:-/data/group_data/rl/geney/craftax_unlabelled_symbolic}"
NUM_WORKERS="${NUM_WORKERS:-16}"

REDIS_PORT=6379
HOSTNAME_FILE="/data/group_data/rl/geney/redis_host.txt"
REDIS_DIR="/home/geney/redis-stable"
QUEUE_NAME="craftax_llm_job_queue"
BASELINES_DIR="/home/geney/Craftax_Baselines"

echo "=========================================="
echo "UNIFIED LLM LABELLING (WITH HIDDEN STATES)"
echo "=========================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Source directory: $SOURCE_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "  Number of workers: $NUM_WORKERS"
echo "=========================================="

# =============================================================================
# STEP 1: Setup environment
# =============================================================================
echo ""
echo "[1/6] Setting up environment..."
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/imaug

# Create directories if needed
mkdir -p "$RESULTS_DIR"
mkdir -p "$BASELINES_DIR/logs"

# =============================================================================
# STEP 2: Start Redis
# =============================================================================
echo ""
echo "[2/6] Starting Redis on $(hostname):${REDIS_PORT}..."
cd "$REDIS_DIR"

# Kill any existing Redis on this port (in case of restart)
./src/redis-cli -p $REDIS_PORT SHUTDOWN NOSAVE 2>/dev/null || true
sleep 2

# Start Redis
./src/redis-server --port $REDIS_PORT --bind 0.0.0.0 --protected-mode no --daemonize yes
sleep 2

# Verify Redis is running
./src/redis-cli -p $REDIS_PORT ping || { echo "ERROR: Redis failed to start!"; exit 1; }
echo "Redis running!"

# Write hostname for workers to find
echo "$(hostname)" > "$HOSTNAME_FILE"
echo "Wrote hostname to $HOSTNAME_FILE"

# =============================================================================
# STEP 3: Clear stale queue and add pending jobs
# =============================================================================
echo ""
echo "[3/6] Queueing pending jobs..."

# Clear any stale entries
./src/redis-cli -p $REDIS_PORT DEL "$QUEUE_NAME" >/dev/null

# Run the queueing script
cd "$BASELINES_DIR/labelling"
python addtoqueue_llm.py --host $(hostname) --symbolic --queue "$QUEUE_NAME"

# Get queue size
QUEUE_SIZE=$(redis-cli -h $(hostname) -p $REDIS_PORT LLEN "$QUEUE_NAME")
echo "Queued $QUEUE_SIZE jobs"

if [ "$QUEUE_SIZE" -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "ALL FILES ALREADY COMPLETED!"
    echo "=========================================="
    redis-cli -h $(hostname) -p $REDIS_PORT SHUTDOWN NOSAVE || true
    exit 0
fi

# =============================================================================
# STEP 4: Submit GPU workers (using llm_worker.py WITH HIDDEN STATES)
# =============================================================================
echo ""
echo "[4/6] Submitting $NUM_WORKERS GPU workers..."

# Calculate workers needed (don't spawn more than queue size)
ACTUAL_WORKERS=$((QUEUE_SIZE < NUM_WORKERS ? QUEUE_SIZE : NUM_WORKERS))
echo "  Spawning $ACTUAL_WORKERS workers for $QUEUE_SIZE jobs"

# Submit worker array job using llm_worker.py (NOT vllm_labeller!)
WORKER_JOB_ID=$(sbatch --parsable --array=1-${ACTUAL_WORKERS} "$BASELINES_DIR/labelling/makeworkers_llm.sbatch")
echo "  Submitted worker array job: $WORKER_JOB_ID"

# =============================================================================
# STEP 5: Monitor until complete
# =============================================================================
echo ""
echo "[5/6] Monitoring progress..."
echo "Queue will be checked every 60 seconds"
echo "Will wait for all workers to finish after queue empties"
echo ""

ZERO_COUNT=0
ZERO_WAIT_CYCLES=10  # Wait 10 more minutes after queue empties

while true; do
    REMAINING=$(redis-cli -h $(hostname) -p $REDIS_PORT LLEN "$QUEUE_NAME")
    COMPLETED=$(ls "$RESULTS_DIR"/*.npz 2>/dev/null | wc -l)
    
    # Check if any workers are still running
    ACTIVE_WORKERS=$(squeue -u $USER -n llm_worker -h 2>/dev/null | wc -l)
    
    echo "$(date '+%Y-%m-%d %H:%M:%S'): Queue=$REMAINING | Completed=$COMPLETED | Workers=$ACTIVE_WORKERS"
    
    if [ "$REMAINING" -eq 0 ]; then
        ZERO_COUNT=$((ZERO_COUNT + 1))
        
        if [ "$ACTIVE_WORKERS" -eq 0 ]; then
            echo ""
            echo "All jobs completed and all workers finished!"
            break
        fi
        
        if [ "$ZERO_COUNT" -ge "$ZERO_WAIT_CYCLES" ]; then
            echo "WARNING: Queue empty for $ZERO_COUNT cycles but $ACTIVE_WORKERS workers still running"
            echo "         Workers may be processing their final jobs..."
        fi
    else
        ZERO_COUNT=0
    fi
    
    sleep 60
done

# =============================================================================
# STEP 6: Cleanup and summary
# =============================================================================
echo ""
echo "[6/6] Cleanup..."

redis-cli -h $(hostname) -p $REDIS_PORT SHUTDOWN NOSAVE || true

# Final count
FINAL_COUNT=$(ls "$RESULTS_DIR"/*.npz 2>/dev/null | wc -l)
SOURCE_COUNT=$(ls "$SOURCE_DIR"/*.npz 2>/dev/null | wc -l)

echo ""
echo "=========================================="
echo "LABELLING COMPLETE"
echo "=========================================="
echo "Date: $(date)"
echo "Total source files: $SOURCE_COUNT"
echo "Total completed: $FINAL_COUNT"
if [ "$FINAL_COUNT" -eq "$SOURCE_COUNT" ]; then
    echo "Status: ALL FILES PROCESSED ✓"
else
    MISSING=$((SOURCE_COUNT - FINAL_COUNT))
    echo "Status: $MISSING files remaining (run script again to resume)"
fi
echo "=========================================="

# Verify hidden states exist in output
echo ""
echo "Verifying hidden states in output..."
SAMPLE_FILE=$(ls "$RESULTS_DIR"/*.npz 2>/dev/null | tail -1)
if [ -n "$SAMPLE_FILE" ]; then
    python -c "
import numpy as np
f = np.load('$SAMPLE_FILE', allow_pickle=True)
print('Keys in output:', list(f.keys()))
if 'hidden_state' in f.keys():
    print('✓ Hidden states present! Shape:', f['hidden_state'].shape)
else:
    print('✗ WARNING: Hidden states missing!')
"
fi
