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
# Takes vLLM outputs (with text_generated but no hidden_state) and adds
# hidden states via fast prefill (10-30x faster than regeneration).
#
# Usage:
#   sbatch labelling/run_extract_hidden.sbatch
#
# Input: Files in craftax_llm_labelled_results (vLLM outputs)
# Output: Files in new_craftax_llm_labelled_results (with hidden_state added)
###############################################################################

set -e

INPUT_DIR="${INPUT_DIR:-/data/group_data/rl/geney/craftax_llm_labelled_results}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/group_data/rl/geney/new_craftax_llm_labelled_results}"
NUM_WORKERS="${NUM_WORKERS:-64}"

REDIS_PORT=6379
HOSTNAME_FILE="/data/group_data/rl/geney/redis_host.txt"
REDIS_DIR="/home/geney/redis-stable"
QUEUE_NAME="craftax_hidden_extraction_queue"
BASELINES_DIR="/home/geney/Craftax_Baselines"

echo "=========================================="
echo "HIDDEN STATE EXTRACTION (PREFILL MODE)"
echo "=========================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/imaug

mkdir -p "$OUTPUT_DIR"
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
echo "[2/5] Scanning for files needing extraction..."

# Clear queue
./src/redis-cli -p $REDIS_PORT DEL "$QUEUE_NAME" >/dev/null

# Find input files that don't have corresponding output
QUEUED=0
for f in "$INPUT_DIR"/*.npz; do
    basename=$(basename "$f")
    output_file="$OUTPUT_DIR/$basename"
    
    if [ ! -f "$output_file" ]; then
        ./src/redis-cli -p $REDIS_PORT LPUSH "$QUEUE_NAME" "$f" >/dev/null
        QUEUED=$((QUEUED + 1))
    fi
done

echo "Queued $QUEUED files for extraction"

if [ "$QUEUED" -eq 0 ]; then
    echo "All files already processed!"
    redis-cli -p $REDIS_PORT SHUTDOWN NOSAVE || true
    exit 0
fi

# =============================================================================
# Submit GPU workers
# =============================================================================
echo ""
echo "[3/5] Submitting $NUM_WORKERS GPU workers..."

ACTUAL_WORKERS=$((QUEUED < NUM_WORKERS ? QUEUED : NUM_WORKERS))
WORKER_JOB_ID=$(sbatch --parsable --array=1-${ACTUAL_WORKERS} "$BASELINES_DIR/labelling/makeworkers_extract.sbatch")
echo "  Submitted job array: $WORKER_JOB_ID"

# =============================================================================
# Monitor
# =============================================================================
echo ""
echo "[4/5] Monitoring progress..."

while true; do
    REMAINING=$(redis-cli -h $(hostname) -p $REDIS_PORT LLEN "$QUEUE_NAME")
    COMPLETED=$(ls "$OUTPUT_DIR"/*.npz 2>/dev/null | wc -l)
    ACTIVE_WORKERS=$(squeue -u $USER -n extract_hidden -h 2>/dev/null | wc -l)
    
    echo "$(date '+%H:%M:%S'): Queue=$REMAINING | Completed=$COMPLETED | Workers=$ACTIVE_WORKERS"
    
    if [ "$REMAINING" -eq 0 ] && [ "$ACTIVE_WORKERS" -eq 0 ]; then
        echo "All done!"
        break
    fi
    
    sleep 30
done

# =============================================================================
# Cleanup and verify
# =============================================================================
echo ""
echo "[5/5] Cleanup and verification..."

redis-cli -h $(hostname) -p $REDIS_PORT SHUTDOWN NOSAVE || true

FINAL_COUNT=$(ls "$OUTPUT_DIR"/*.npz 2>/dev/null | wc -l)
INPUT_COUNT=$(ls "$INPUT_DIR"/*.npz 2>/dev/null | wc -l)

echo ""
echo "=========================================="
echo "EXTRACTION COMPLETE"
echo "=========================================="
echo "Input files:  $INPUT_COUNT"
echo "Output files: $FINAL_COUNT"

# Verify hidden states in a sample file
SAMPLE=$(ls "$OUTPUT_DIR"/*.npz 2>/dev/null | tail -1)
if [ -n "$SAMPLE" ]; then
    python -c "
import numpy as np
f = np.load('$SAMPLE', allow_pickle=True)
print('Keys:', list(f.keys()))
if 'hidden_state' in f.keys():
    print('✓ hidden_state shape:', f['hidden_state'].shape)
else:
    print('✗ hidden_state MISSING!')
"
fi
echo "=========================================="
