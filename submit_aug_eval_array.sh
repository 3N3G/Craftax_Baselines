#!/bin/bash
#SBATCH --job-name=craftax_eval_array
#SBATCH --output=logs/eval_array_%A_%a.out  # %A=JobId, %a=ArrayTaskId
#SBATCH --error=logs/eval_array_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=rl
#SBATCH --gres=gpu:1        # Request 1 GPU per Server/Client pair
#SBATCH --cpus-per-task=8   # Ensure enough CPU cores for JAX/Data loading
#SBATCH --mem=64G

# ==========================================
# 1. INPUT ARGUMENTS
# Usage: sbatch --array=1-M submit_eval_array.sh [CHECKPOINT_PATH] [N_EPISODES]
# ==========================================
CHECKPOINT_PATH="${1:-/data/group_data/rl/geney/checkpoints/awr_augmented/awr_aug_checkpoint_100000.pth}"
NUM_EPISODES="${2:-3}" # Default to 3 if not provided

# Derived Paths
STATS_PATH="/data/group_data/rl/geney/checkpoints/awr_augmented/hidden_state_stats.npz"
REPO_DIR="/home/geney/Craftax_Baselines"
CONDA_ENV="/data/user_data/geney/.conda/envs/test"

# ==========================================
# 2. DYNAMIC NETWORKING SETUP
# ==========================================
# We run Server and Client on the same node.
# We calculate a unique port based on the Task ID to prevent collisions
# if multiple tasks somehow land on the same node (unlikely with gpu:1, but safe).
BASE_PORT=5000
SERVER_PORT=$((BASE_PORT + SLURM_ARRAY_TASK_ID))
SERVER_URL="http://127.0.0.1:${SERVER_PORT}"

# ==========================================
# 3. ENVIRONMENT SETUP
# ==========================================
source ~/.bashrc
conda activate "$CONDA_ENV"
cd "$REPO_DIR" || exit 1

export PYTHONUNBUFFERED=1
export WANDB_API_KEY="e63622f162f8704ad6a057b040097fa72d663ef6"
export WANDB_DIR=$(mktemp -d)

# Cleanup Function: Ensures Server is killed when Client finishes or job is cancelled
cleanup() {
    echo "Stopping VLM Server (PID: $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null
    rm -rf "$WANDB_DIR"
}
trap cleanup EXIT

# ==========================================
# 4. START VLM SERVER (BACKGROUND)
# ==========================================
echo ">>> [Task ${SLURM_ARRAY_TASK_ID}] Starting VLM Server on Port $SERVER_PORT..."

# Note: We do NOT export JAX_PLATFORMS="cpu" yet, so the Server finds the GPU.
python vlm_server.py --port "$SERVER_PORT" > "logs/server_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
SERVER_PID=$!

# Wait for Server to be ready
echo ">>> Waiting for server to initialize at $SERVER_URL..."
MAX_RETRIES=60 # Wait up to ~5 minutes (60 * 5s)
COUNT=0
while ! curl -s "$SERVER_URL" > /dev/null; do
    if [ "$COUNT" -ge "$MAX_RETRIES" ]; then
        echo "Error: Server failed to start within time limit."
        cat "logs/server_${SLURM_ARRAY_TASK_ID}.log"
        exit 1
    fi
    sleep 5
    ((COUNT++))
    # Check if process died
    if ! ps -p "$SERVER_PID" > /dev/null; then
         echo "Error: Server process died unexpectedly."
         cat "logs/server_${SLURM_ARRAY_TASK_ID}.log"
         exit 1
    fi
    echo "Waiting... ($COUNT/$MAX_RETRIES)"
done

echo ">>> Server is UP and RUNNING."

# ==========================================
# 5. START CLIENT (FOREGROUND)
# ==========================================
echo ">>> Starting Client Evaluation..."

# Now we set JAX to CPU for the client (as per your original script)
export JAX_PLATFORMS="cpu"
export JAX_PLATFORM_NAME="cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

# Create a unique video directory for this array task so they don't overwrite
VIDEO_OUTPUT_DIR="./eval_videos_proper/batch_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$VIDEO_OUTPUT_DIR"

python eval_awr_aug_client_proper.py \
    --checkpoint="$CHECKPOINT_PATH" \
    --server_url="$SERVER_URL" \
    --num_episodes "$NUM_EPISODES" \
    --save_video \
    --video_dir "$VIDEO_OUTPUT_DIR" \
    --stats_path "$STATS_PATH"

CLIENT_EXIT_CODE=$?

if [ $CLIENT_EXIT_CODE -eq 0 ]; then
    echo ">>> [Task ${SLURM_ARRAY_TASK_ID}] Evaluation Success!"
else
    echo ">>> [Task ${SLURM_ARRAY_TASK_ID}] Evaluation Failed!"
fi

exit $CLIENT_EXIT_CODE
# Trap will automatically kill the server now.
