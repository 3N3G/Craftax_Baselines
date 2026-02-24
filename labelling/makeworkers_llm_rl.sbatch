#!/bin/bash

#SBATCH --job-name=llm_worker
#SBATCH --partition=rl
#SBATCH --array=1-30                   # Default, can be overridden with --array=1-N
#SBATCH --output=/home/geney/Craftax_Baselines/logs/llm_worker_%A_%a.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/llm_worker_%A_%a.err
#
# --- Resource Request ---
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#
# --- Optional ---
#SBATCH --time=24:00:00
#SBATCH --mem=64G

set -eo pipefail

echo "Starting LLM worker on node $(hostname)"

ENV_PATH_DEFAULT="/data/user_data/geney/.conda/envs/craftax_fast_llm"
ENV_BIN_DEFAULT="${ENV_PATH_DEFAULT}/bin"

# Prefer direct env bin path to avoid shell init fragility on compute nodes.
if [[ -d "${ENV_BIN_DEFAULT}" ]]; then
    export PATH="${ENV_BIN_DEFAULT}:${PATH}"
fi

if ! command -v python >/dev/null 2>&1 || ! command -v vllm >/dev/null 2>&1; then
    if [[ -f "${HOME}/.bashrc" ]]; then
        # Best-effort only; some cluster /etc/bashrc variants return non-zero in batch mode.
        source "${HOME}/.bashrc" >/dev/null 2>&1 || true
    fi
    if command -v conda >/dev/null 2>&1; then
        conda activate "${ENV_PATH_DEFAULT}" >/dev/null 2>&1 || conda activate craftax_fast_llm >/dev/null 2>&1 || true
    fi
fi

if ! command -v python >/dev/null 2>&1 || ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: required runtime tools not found."
    echo "  python path: $(command -v python || echo missing)"
    echo "  vllm path: $(command -v vllm || echo missing)"
    exit 1
fi

echo "Runtime python: $(command -v python)"
echo "Runtime vllm: $(command -v vllm)"
python -V || true

cd /home/geney/Craftax_Baselines

# Per-worker runtime dirs on node-local scratch to reduce NAS metadata load.
MIN_LOCAL_FREE_GB="${MIN_LOCAL_FREE_GB:-50}"
HARD_MIN_LOCAL_FREE_GB="${HARD_MIN_LOCAL_FREE_GB:-25}"

free_gb() {
    local path="$1"
    df -BG --output=avail "$path" 2>/dev/null | tail -1 | tr -dc '0-9'
}

LOCAL_BASE=""
BEST_FREE_GB=-1
for candidate in "/scratch/${USER}" "/tmp/${USER}"; do
    mkdir -p "${candidate}" 2>/dev/null || true
    if [[ -d "${candidate}" && -w "${candidate}" ]]; then
        candidate_free_gb="$(free_gb "${candidate}")"
        if [[ -n "${candidate_free_gb}" && "${candidate_free_gb}" -gt "${BEST_FREE_GB}" ]]; then
            LOCAL_BASE="${candidate}"
            BEST_FREE_GB="${candidate_free_gb}"
        fi
    fi
done

if [[ -z "${LOCAL_BASE}" ]]; then
    echo "ERROR: no writable local scratch path under /scratch/${USER} or /tmp/${USER}."
    exit 1
fi

WORKER_TMP_ROOT="${LOCAL_BASE}/craftax_worker_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${WORKER_TMP_ROOT}/tmp" "${WORKER_TMP_ROOT}/wandb" "${WORKER_TMP_ROOT}/wandb_cache" "${WORKER_TMP_ROOT}/wandb_config"
export TMPDIR="${WORKER_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export WORKER_LOCAL_DIR="${WORKER_TMP_ROOT}/work"
export VLLM_TMP_ROOT="${WORKER_TMP_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${WORKER_TMP_ROOT}/vllm/torchinductor"
export XDG_CACHE_HOME="${WORKER_TMP_ROOT}/.cache"
export WANDB_DIR="${WORKER_TMP_ROOT}/wandb"
export WANDB_CACHE_DIR="${WORKER_TMP_ROOT}/wandb_cache"
export WANDB_CONFIG_DIR="${WORKER_TMP_ROOT}/wandb_config"
export FLUSH_EVERY_BATCHES="${FLUSH_EVERY_BATCHES:-8}"
export WANDB_LOG_EVERY_BATCHES="${WANDB_LOG_EVERY_BATCHES:-16}"
export PROGRESS_LOG_EVERY_BATCHES="${PROGRESS_LOG_EVERY_BATCHES:-16}"
export WANDB_PER_FILE="${WANDB_PER_FILE:-0}"
export MAX_JOB_RETRIES="${MAX_JOB_RETRIES:-2}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_PORT_BASE="${VLLM_PORT_BASE:-18000}"

mkdir -p "${WORKER_LOCAL_DIR}" "${VLLM_TMP_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}"

WORKER_FREE_GB="$(free_gb "${WORKER_TMP_ROOT}")"
echo "Local worker root: ${WORKER_TMP_ROOT}"
echo "Free local space: ${WORKER_FREE_GB:-unknown}G (target ${MIN_LOCAL_FREE_GB}G, hard minimum ${HARD_MIN_LOCAL_FREE_GB}G)"
df -h "${WORKER_TMP_ROOT}" || true
df -i "${WORKER_TMP_ROOT}" || true

if [[ -n "${WORKER_FREE_GB}" && "${WORKER_FREE_GB}" -lt "${HARD_MIN_LOCAL_FREE_GB}" ]]; then
    echo "ERROR: insufficient local space at ${WORKER_TMP_ROOT} (below hard minimum)."
    exit 1
fi
if [[ -n "${WORKER_FREE_GB}" && "${WORKER_FREE_GB}" -lt "${MIN_LOCAL_FREE_GB}" ]]; then
    echo "WARNING: local space below target threshold; continuing."
fi

VLLM_GPU_MEMORY_UTILIZATION_DEFAULT="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
VLLM_GPU_MEMORY_UTILIZATION_SAFE="${VLLM_GPU_MEMORY_UTILIZATION_SAFE:-0.50}"
VLLM_GPU_MEMORY_UTILIZATION_LOW="${VLLM_GPU_MEMORY_UTILIZATION_LOW:-0.40}"
VLLM_LOG_PATH="${WORKER_TMP_ROOT}/vllm_worker.log"

find_open_port() {
    local port="$1"
    local tries=0
    while [[ $tries -lt 64 ]]; do
        if ! ss -ltn "( sport = :${port} )" 2>/dev/null | grep -q ":${port}"; then
            echo "${port}"
            return 0
        fi
        port=$((port + 1))
        tries=$((tries + 1))
    done
    return 1
}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]]; then
    TASK_ID=0
fi
DESIRED_PORT=$((VLLM_PORT_BASE + TASK_ID))
VLLM_PORT="$(find_open_port "${DESIRED_PORT}")"
if [[ -z "${VLLM_PORT}" ]]; then
    echo "ERROR: failed to find free port near ${DESIRED_PORT}"
    exit 1
fi
export VLLM_PORT
export VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
echo "Using VLLM_URL=${VLLM_URL}"

start_vllm_server() {
    local safe_flag="$1"
    local gpu_util="$2"
    export VLLM_GPU_MEMORY_UTILIZATION="${gpu_util}"
    echo "Launching vLLM (safe_flag='${safe_flag}', gpu_util=${VLLM_GPU_MEMORY_UTILIZATION})"
    bash scripts/start_vllm_hidden.sh --mode last_token --port "${VLLM_PORT}" ${safe_flag} > "${VLLM_LOG_PATH}" 2>&1 &
    VLLM_PID=$!
}

stop_vllm_server() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        unset VLLM_PID
    fi
}

wait_for_vllm_health() {
    local timeout="$1"
    for i in $(seq 1 "${timeout}"); do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "vLLM process exited early."
            return 1
        fi
        if curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
            # Verify we're talking to the expected served model, not another
            # process already listening on this node.
            if curl -s "${VLLM_URL}/v1/models" | grep -q '\./configs/vllm_hidden_qwen4b'; then
                echo "vLLM server ready at ${VLLM_URL}"
                return 0
            fi
            echo "Port ${VLLM_PORT} is serving an unexpected model; retrying..."
        fi
        sleep 1
    done
    return 1
}

start_attempt() {
    local label="$1"
    local safe_flag="$2"
    local gpu_util="$3"
    local timeout="$4"

    echo "Starting vLLM server (${label})..."
    start_vllm_server "${safe_flag}" "${gpu_util}"
    echo "Waiting for vLLM server (${label})..."
    if wait_for_vllm_health "${timeout}"; then
        echo "vLLM startup succeeded (${label})."
        return 0
    fi
    echo "vLLM startup failed (${label}); cleaning up and retrying..."
    stop_vllm_server
    rm -rf "${TORCHINDUCTOR_CACHE_DIR:?}/"* || true
    return 1
}

# Start vLLM server on this node (each worker has its own GPU + server)
if ! start_attempt "standard" "" "${VLLM_GPU_MEMORY_UTILIZATION_DEFAULT}" 180 \
    && ! start_attempt "safe-eager" "--safe" "${VLLM_GPU_MEMORY_UTILIZATION_SAFE}" 180 \
    && ! start_attempt "safe-eager-low-mem" "--safe" "${VLLM_GPU_MEMORY_UTILIZATION_LOW}" 240; then
    echo "ERROR: vLLM server failed to start after all retries"
    tail -n 200 "${VLLM_LOG_PATH}" || true
    exit 1
fi

# Run the worker
cd /home/geney/Craftax_Baselines/labelling
srun --cpu-bind=none python llm_worker.py

# Cleanup
echo "Stopping vLLM server..."
stop_vllm_server
rm -rf "${WORKER_TMP_ROOT}" || true
echo "Worker done."
