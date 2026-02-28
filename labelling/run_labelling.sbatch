#!/bin/bash
#SBATCH --job-name=llm_label_all
#SBATCH --partition=cpu
#SBATCH --output=/home/geney/Craftax_Baselines/logs/label_all_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/label_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120

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
#   ls /data/group_data/rl/geney/vllm_craftax_labelled_results/*.npz | wc -l
#
# Optional environment variables (set before sbatch or in script):
#   RESULTS_DIR - Where to save results (default: /data/group_data/rl/geney/vllm_craftax_labelled_results)
#   SOURCE_DIR  - Where to find input files (default: /data/group_data/rl/geney/craftax_unlabelled_symbolic)
#   NUM_WORKERS - Number of GPU workers to spawn (default: 16)
###############################################################################

set -e

# =============================================================================
# CONFIGURATION - Edit these or override via environment variables
# =============================================================================
RESULTS_DIR="${RESULTS_DIR:-/data/group_data/rl/geney/vllm_craftax_labelled_results}"
SOURCE_DIR="${SOURCE_DIR:-/data/group_data/rl/geney/craftax_unlabelled_symbolic}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-120}"
RESULTS_PATTERN="${RESULTS_PATTERN:-trajectories_batch_*.npz}"
STALL_CYCLES="${STALL_CYCLES:-6}"
TOPUP_COOLDOWN_SEC="${TOPUP_COOLDOWN_SEC:-300}"

REDIS_PORT=6379
HOSTNAME_FILE="/data/group_data/rl/geney/redis_host.txt"
REDIS_DIR="/home/geney/redis-stable"
QUEUE_NAME="craftax_llm_job_queue"
RETRY_HASH_NAME="craftax_llm_job_retry_counts"
COMPLETED_COUNTER_NAME="craftax_llm_completed_count"
BASELINES_DIR="/home/geney/Craftax_Baselines"
SUBMITTED_WORKER_JOBS=()
CLEANUP_DONE=0
LAST_TOPUP_EPOCH=0

count_workers_by_state() {
    local states="$1"
    local total=0
    local jid=""
    for jid in "${SUBMITTED_WORKER_JOBS[@]}"; do
        if [ -n "$jid" ]; then
            local c
            c=$(squeue -h -j "$jid" -t "$states" 2>/dev/null | wc -l)
            total=$((total + c))
        fi
    done
    echo "$total"
}

submit_worker_array() {
    local worker_count="$1"
    local label="$2"
    if [ "$worker_count" -le 0 ]; then
        return 0
    fi
    local jid
    jid=$(sbatch --parsable --array=1-"${worker_count}" "$BASELINES_DIR/labelling/makeworkers_llm.sbatch")
    SUBMITTED_WORKER_JOBS+=("$jid")
    LAST_TOPUP_EPOCH=$(date +%s)
    echo "  Submitted ${label} worker array job: $jid (${worker_count} tasks)"
}

cleanup() {
    if [ "$CLEANUP_DONE" -eq 1 ]; then
        return
    fi
    CLEANUP_DONE=1

    echo ""
    echo "[cleanup] Triggered at $(date)"
    if [ "${#SUBMITTED_WORKER_JOBS[@]}" -gt 0 ]; then
        echo "[cleanup] Cancelling worker arrays: ${SUBMITTED_WORKER_JOBS[*]}"
        scancel "${SUBMITTED_WORKER_JOBS[@]}" 2>/dev/null || true
    fi

    if [ -x "${REDIS_DIR}/src/redis-cli" ]; then
        ${REDIS_DIR}/src/redis-cli -h "$(hostname)" -p "${REDIS_PORT}" SHUTDOWN NOSAVE >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT INT TERM

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
echo "  Monitor interval: ${MONITOR_INTERVAL}s"
echo "=========================================="

# =============================================================================
# STEP 1: Setup environment
# =============================================================================
echo ""
echo "[1/6] Setting up environment..."
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/craftax_fast_llm

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
./src/redis-cli -p $REDIS_PORT DEL "$RETRY_HASH_NAME" >/dev/null
./src/redis-cli -p $REDIS_PORT DEL "$COMPLETED_COUNTER_NAME" >/dev/null

# Run the queueing script
cd "$BASELINES_DIR/labelling"
python addtoqueue_llm.py --host $(hostname) --symbolic --queue "$QUEUE_NAME"

# Get queue size
QUEUE_SIZE=$($REDIS_DIR/src/redis-cli -h $(hostname) -p $REDIS_PORT LLEN "$QUEUE_NAME")
INITIAL_COMPLETED=$(find "$RESULTS_DIR" -maxdepth 1 -name "$RESULTS_PATTERN" 2>/dev/null | wc -l)
$REDIS_DIR/src/redis-cli -h $(hostname) -p $REDIS_PORT SET "$COMPLETED_COUNTER_NAME" "$INITIAL_COMPLETED" >/dev/null
echo "Queued $QUEUE_SIZE jobs"
echo "Initialized completed counter: $INITIAL_COMPLETED"

if [ "$QUEUE_SIZE" -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "ALL FILES ALREADY COMPLETED!"
    echo "=========================================="
    $REDIS_DIR/src/redis-cli -h $(hostname) -p $REDIS_PORT SHUTDOWN NOSAVE || true
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
submit_worker_array "$ACTUAL_WORKERS" "initial"

# =============================================================================
# STEP 5: Monitor, detect lost jobs, re-queue, repeat
# =============================================================================
echo ""
echo "[5/6] Monitoring progress (with automatic re-queue on worker failure)..."
echo "Queue will be checked every ${MONITOR_INTERVAL} seconds"
echo ""

SOURCE_COUNT=$(find "$SOURCE_DIR" -maxdepth 1 -name "*.npz" 2>/dev/null | wc -l)
ROUND=1

while true; do
    # --- Inner monitoring loop: wait for queue to drain and workers to finish ---
    ZERO_COUNT=0
    STALLED_COUNT=0
    while true; do
        REMAINING=$($REDIS_DIR/src/redis-cli -h $(hostname) -p $REDIS_PORT LLEN "$QUEUE_NAME")
        # Use filesystem counts as source of truth; Redis increments can drift
        # upward due retries or duplicate completions.
        COMPLETED=$(find "$RESULTS_DIR" -maxdepth 1 -name "$RESULTS_PATTERN" 2>/dev/null | wc -l)
        $REDIS_DIR/src/redis-cli -h $(hostname) -p $REDIS_PORT SET "$COMPLETED_COUNTER_NAME" "$COMPLETED" >/dev/null
        ACTIVE_WORKERS=$(count_workers_by_state R)
        PENDING_WORKERS=$(count_workers_by_state PD)
        TOTAL_WORKERS=$((ACTIVE_WORKERS + PENDING_WORKERS))
        TARGET_WORKERS=$((REMAINING < NUM_WORKERS ? REMAINING : NUM_WORKERS))
        if [ "$TARGET_WORKERS" -lt 1 ]; then
            TARGET_WORKERS=1
        fi

        NOW_EPOCH=$(date +%s)
        SINCE_TOPUP=$((NOW_EPOCH - LAST_TOPUP_EPOCH))

        # Top-up only when this run has no pending tasks; otherwise let Slurm schedule
        # already-submitted workers instead of creating queue storms.
        if [ "$REMAINING" -gt 0 ] && [ "$TOTAL_WORKERS" -lt "$TARGET_WORKERS" ] && [ "$PENDING_WORKERS" -eq 0 ] && [ "$SINCE_TOPUP" -ge "$TOPUP_COOLDOWN_SEC" ]; then
            SHORTFALL=$((TARGET_WORKERS - TOTAL_WORKERS))
            if [ "$SHORTFALL" -gt 0 ]; then
                echo "  Worker shortfall detected (running=$ACTIVE_WORKERS pending=$PENDING_WORKERS target=$TARGET_WORKERS)."
                echo "  Submitting top-up array of $SHORTFALL workers..."
                submit_worker_array "$SHORTFALL" "top-up"
                PENDING_WORKERS=$(count_workers_by_state PD)
                TOTAL_WORKERS=$((ACTIVE_WORKERS + PENDING_WORKERS))
            fi
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') [Round $ROUND]: Queue=$REMAINING | Completed=$COMPLETED/$SOURCE_COUNT | Workers=${ACTIVE_WORKERS}R/${PENDING_WORKERS}PD"

        if [ "$REMAINING" -eq 0 ] && [ "$ACTIVE_WORKERS" -eq 0 ]; then
            echo ""
            echo "Queue empty and all workers finished."
            break
        fi

        # Workers can die while queue still has jobs. Only mark stalled when there are
        # no running AND no pending workers for this run.
        if [ "$REMAINING" -gt 0 ] && [ "$ACTIVE_WORKERS" -eq 0 ] && [ "$PENDING_WORKERS" -eq 0 ]; then
            STALLED_COUNT=$((STALLED_COUNT + 1))
            if [ "$STALLED_COUNT" -ge "$STALL_CYCLES" ]; then
                STALLED_SECONDS=$((STALLED_COUNT * MONITOR_INTERVAL))
                echo ""
                echo "Detected stalled queue (remaining=$REMAINING, workers=0R/0PD) for ${STALLED_SECONDS}s."
                echo "Breaking monitor loop to re-queue and relaunch workers."
                break
            fi
        else
            STALLED_COUNT=0
        fi

        if [ "$REMAINING" -eq 0 ]; then
            ZERO_COUNT=$((ZERO_COUNT + 1))
            if [ "$ZERO_COUNT" -ge 5 ] && [ "$ACTIVE_WORKERS" -gt 0 ]; then
                ZERO_SECONDS=$((ZERO_COUNT * MONITOR_INTERVAL))
                echo "  (Queue empty for ${ZERO_SECONDS}s, $ACTIVE_WORKERS workers still processing...)"
            fi
        else
            ZERO_COUNT=0
        fi

        sleep "$MONITOR_INTERVAL"
    done

    # --- Check: are we done? ---
    COMPLETED=$(find "$RESULTS_DIR" -maxdepth 1 -name "$RESULTS_PATTERN" 2>/dev/null | wc -l)
    $REDIS_DIR/src/redis-cli -h $(hostname) -p $REDIS_PORT SET "$COMPLETED_COUNTER_NAME" "$COMPLETED" >/dev/null
    if [ "$COMPLETED" -ge "$SOURCE_COUNT" ]; then
        echo ""
        echo "✓ All $SOURCE_COUNT files completed!"
        break
    fi

    # --- Find missing jobs and re-queue ---
    MISSING=$((SOURCE_COUNT - COMPLETED))
    echo ""
    echo "=========================================="
    echo "INCOMPLETE: $COMPLETED/$SOURCE_COUNT done ($MISSING missing)"
    echo "Re-queuing missing jobs (Round $((ROUND + 1)))..."
    echo "=========================================="

    # Clear queue and re-add only missing files
    $REDIS_DIR/src/redis-cli -h $(hostname) -p $REDIS_PORT DEL "$QUEUE_NAME" > /dev/null
    $REDIS_DIR/src/redis-cli -h $(hostname) -p $REDIS_PORT DEL "$RETRY_HASH_NAME" > /dev/null

    # Get list of completed basenames
    COMPLETED_SET=$(ls "$RESULTS_DIR"/$RESULTS_PATTERN 2>/dev/null | xargs -I{} basename {})

    # Re-queue source files that don't have a completed output
    REQUEUED=0
    for src_file in "$SOURCE_DIR"/*.npz; do
        base=$(basename "$src_file")
        if ! echo "$COMPLETED_SET" | grep -qx "$base"; then
            $REDIS_DIR/src/redis-cli -h $(hostname) -p $REDIS_PORT LPUSH "$QUEUE_NAME" "$src_file" > /dev/null
            REQUEUED=$((REQUEUED + 1))
        fi
    done
    echo "  Re-queued $REQUEUED jobs"

    if [ "$REQUEUED" -eq 0 ]; then
        echo "  Nothing to re-queue — all files present!"
        break
    fi

    # Spawn new workers
    NEW_WORKERS=$((REQUEUED < NUM_WORKERS ? REQUEUED : NUM_WORKERS))
    echo "  Spawning $NEW_WORKERS new workers..."
    submit_worker_array "$NEW_WORKERS" "requeue"

    ROUND=$((ROUND + 1))
    echo ""
done

# =============================================================================
# STEP 6: Cleanup and summary
# =============================================================================
echo ""
echo "[6/6] Cleanup..."

$REDIS_DIR/src/redis-cli -h $(hostname) -p $REDIS_PORT SHUTDOWN NOSAVE || true

# Final count
FINAL_COUNT=$(find "$RESULTS_DIR" -maxdepth 1 -name "*.npz" 2>/dev/null | wc -l)

echo ""
echo "=========================================="
echo "LABELLING COMPLETE"
echo "=========================================="
echo "Date: $(date)"
echo "Total source files: $SOURCE_COUNT"
echo "Total completed: $FINAL_COUNT"
echo "Rounds needed: $ROUND"
if [ "$FINAL_COUNT" -ge "$SOURCE_COUNT" ]; then
    echo "Status: ALL FILES PROCESSED ✓"
else
    MISSING=$((SOURCE_COUNT - FINAL_COUNT))
    echo "Status: $MISSING files remaining (run script again to resume)"
fi
echo "=========================================="

# Verify hidden states exist in output
echo ""
echo "Verifying hidden states in output..."
SAMPLE_FILE=$(find "$RESULTS_DIR" -maxdepth 1 -name "*.npz" 2>/dev/null | tail -1)
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
