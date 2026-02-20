#!/bin/bash
#SBATCH --job-name=extract_coord
#SBATCH --partition=cpu
#SBATCH --output=/home/geney/Craftax_Baselines/logs/extract_coord_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/extract_coord_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

###############################################################################
# HIDDEN STATE EXTRACTION COORDINATOR
#
# Re-processes existing labelled data (with text_generated but no hidden_state)
# to add hidden states via fast vLLM prefill.
#
# Workers use llm_worker.py which starts its own vLLM server per GPU.
#
# Usage:
#   sbatch labelling/run_extract_hidden.sbatch
#
# Input:  Files in vllm_craftax_labelled_results (with text_generated)
# Output: Same directory, files reprocessed with hidden_state added
###############################################################################

set -e

INPUT_DIR="${INPUT_DIR:-/data/group_data/rl/geney/vllm_craftax_labelled_results}"
NUM_WORKERS="${NUM_WORKERS:-30}"

REDIS_PORT=6379
HOSTNAME_FILE="/data/group_data/rl/geney/redis_host.txt"
REDIS_DIR="/home/geney/redis-stable"
QUEUE_NAME="craftax_llm_job_queue"
BASELINES_DIR="/home/geney/Craftax_Baselines"

echo "=========================================="
echo "HIDDEN STATE EXTRACTION (vLLM PREFILL)"
echo "=========================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""
echo "Input/Output: $INPUT_DIR"
echo "=========================================="

source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/craftax_fast_llm

mkdir -p "$BASELINES_DIR/logs"

# =============================================================================
# Start Redis
# =============================================================================
echo ""
echo "[1/5] Starting Redis..."
cd "$REDIS_DIR"
./src/redis-cli -p $REDIS_PORT SHUTDOWN NOSAVE 2>/dev/null || true
sleep 2
./src/redis-server --port $REDIS_PORT --bind 0.0.0.0 --protected-mode no --daemonize yes
sleep 2
./src/redis-cli -p $REDIS_PORT ping || { echo "ERROR: Redis failed!"; exit 1; }
echo "$(hostname)" > "$HOSTNAME_FILE"
echo "Redis running on $(hostname)"

# =============================================================================
# Queue files that need hidden state extraction
# =============================================================================
echo ""
echo "[2/5] Scanning for files needing hidden state extraction..."

# Clear queue
./src/redis-cli -p $REDIS_PORT DEL "$QUEUE_NAME" >/dev/null

# Find input files that don't have hidden_state
QUEUED=0
for f in "$INPUT_DIR"/*.npz; do
    # Check if file already has hidden_state
    HAS_HIDDEN=$(python3 -c "
import numpy as np
try:
    d = np.load('$f', allow_pickle=True)
    print('yes' if 'hidden_state' in d.files else 'no')
except:
    print('no')
" 2>/dev/null)
    
    if [ "$HAS_HIDDEN" = "no" ]; then
        ./src/redis-cli -p $REDIS_PORT LPUSH "$QUEUE_NAME" "$f" >/dev/null
        QUEUED=$((QUEUED + 1))
    fi
done

echo "Queued $QUEUED files for extraction"

if [ "$QUEUED" -eq 0 ]; then
    echo "All files already have hidden states!"
    ./src/redis-cli -p $REDIS_PORT SHUTDOWN NOSAVE || true
    exit 0
fi

# =============================================================================
# Submit GPU workers (each starts its own vLLM server)
# =============================================================================
echo ""
echo "[3/5] Submitting $NUM_WORKERS GPU workers..."

ACTUAL_WORKERS=$((QUEUED < NUM_WORKERS ? QUEUED : NUM_WORKERS))
WORKER_JOB_ID=$(sbatch --parsable --array=1-${ACTUAL_WORKERS} "$BASELINES_DIR/labelling/makeworkers_llm.sbatch")
echo "  Submitted job array: $WORKER_JOB_ID"

# =============================================================================
# Monitor (self-healing: auto-resubmits workers on preemption)
# =============================================================================
echo ""
echo "[4/5] Monitoring progress (auto-resubmit enabled)..."

ZERO_WORKER_COUNT=0
INPUT_COUNT=$(ls "$INPUT_DIR"/*.npz 2>/dev/null | wc -l)

while true; do
    REMAINING=$(./src/redis-cli -h $(hostname) -p $REDIS_PORT LLEN "$QUEUE_NAME")
    
    # Count files with hidden_state
    COMPLETED=$(python3 -c "
import numpy as np, glob, os
count = 0
for f in glob.glob('$INPUT_DIR/*.npz'):
    try:
        d = np.load(f, allow_pickle=True)
        if 'hidden_state' in d.files:
            count += 1
    except:
        pass
print(count)
" 2>/dev/null)
    
    ACTIVE_WORKERS=$(squeue -u $USER -n llm_worker -h 2>/dev/null | wc -l)
    TRULY_REMAINING=$((INPUT_COUNT - COMPLETED))
    
    echo "$(date '+%H:%M:%S'): Queue=$REMAINING | WithHidden=$COMPLETED/$INPUT_COUNT | Workers=$ACTIVE_WORKERS | Remaining=$TRULY_REMAINING"
    
    # All files completed
    if [ "$TRULY_REMAINING" -le 0 ]; then
        echo "All $INPUT_COUNT files have hidden states!"
        break
    fi
    
    # Workers died but work remains - resubmit!
    if [ "$ACTIVE_WORKERS" -eq 0 ] && [ "$TRULY_REMAINING" -gt 0 ]; then
        ZERO_WORKER_COUNT=$((ZERO_WORKER_COUNT + 1))
        
        # Wait 2 checks (1 min) to confirm workers are truly gone
        if [ "$ZERO_WORKER_COUNT" -ge 2 ]; then
            echo ""
            echo "$(date '+%H:%M:%S'): ⚠️  All workers died! $TRULY_REMAINING files remaining."
            echo "Re-queuing missing files and resubmitting workers..."
            
            # Re-queue files without hidden_state
            ./src/redis-cli -h $(hostname) -p $REDIS_PORT DEL "$QUEUE_NAME" >/dev/null
            REQUEUED=0
            for f in "$INPUT_DIR"/*.npz; do
                HAS_HIDDEN=$(python3 -c "
import numpy as np
try:
    d = np.load('$f', allow_pickle=True)
    print('yes' if 'hidden_state' in d.files else 'no')
except:
    print('no')
" 2>/dev/null)
                if [ "$HAS_HIDDEN" = "no" ]; then
                    ./src/redis-cli -h $(hostname) -p $REDIS_PORT LPUSH "$QUEUE_NAME" "$f" >/dev/null
                    REQUEUED=$((REQUEUED + 1))
                fi
            done
            echo "Re-queued $REQUEUED files"
            
            # Submit new workers
            NEW_WORKERS=$((REQUEUED < NUM_WORKERS ? REQUEUED : NUM_WORKERS))
            WORKER_JOB_ID=$(sbatch --parsable --array=1-${NEW_WORKERS} "$BASELINES_DIR/labelling/makeworkers_llm.sbatch")
            echo "Submitted new worker array: $WORKER_JOB_ID ($NEW_WORKERS workers)"
            echo ""
            
            ZERO_WORKER_COUNT=0
        fi
    else
        ZERO_WORKER_COUNT=0
    fi
    
    sleep 30
done

# =============================================================================
# Cleanup and verify
# =============================================================================
echo ""
echo "[5/5] Cleanup and verification..."

./src/redis-cli -h $(hostname) -p $REDIS_PORT SHUTDOWN NOSAVE || true

# Verify hidden states in a sample file
SAMPLE=$(ls "$INPUT_DIR"/*.npz 2>/dev/null | tail -1)
if [ -n "$SAMPLE" ]; then
    python3 -c "
import numpy as np
f = np.load('$SAMPLE', allow_pickle=True)
print('Keys:', list(f.keys()))
if 'hidden_state' in f.keys():
    hs = f['hidden_state']
    print('✓ hidden_state shape:', hs.shape, 'dtype:', hs.dtype)
else:
    print('✗ hidden_state MISSING!')
"
fi
echo "=========================================="

