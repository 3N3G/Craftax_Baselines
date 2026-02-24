#!/bin/bash
#SBATCH --job-name=llm_finish_once
#SBATCH --partition=cpu
#SBATCH --output=/home/geney/Craftax_Baselines/logs/label_finish_once_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/label_finish_once_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120

set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-/data/group_data/rl/geney/craftax_unlabelled_symbolic}"
RESULTS_DIR="${RESULTS_DIR:-/data/group_data/rl/geney/vllm_craftax_labelled_results}"
RESULTS_PATTERN="${RESULTS_PATTERN:-trajectories_batch_*.npz}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
MAX_ROUNDS="${MAX_ROUNDS:-6}"
MAX_RESCUES_PER_ROUND="${MAX_RESCUES_PER_ROUND:-8}"
RESCUE_COOLDOWN_SEC="${RESCUE_COOLDOWN_SEC:-180}"
LOGS_DIR="${LOGS_DIR:-/scratch/${USER}/craftax_llm_job_logs}"

REDIS_PORT="${REDIS_PORT:-$((6380 + (SLURM_JOB_ID % 500)))}"
REDIS_HOST="$(hostname)"
REDIS_DIR="/home/geney/redis-stable"
BASELINES_DIR="/home/geney/Craftax_Baselines"
VLLM_PORT_BASE="${VLLM_PORT_BASE:-$((18000 + (SLURM_JOB_ID % 2000)))}"
WORKER_SCRIPT="${WORKER_SCRIPT:-${BASELINES_DIR}/labelling/makeworkers_llm.sbatch}"

QUEUE_NAME="craftax_llm_finish_queue_${SLURM_JOB_ID}"
RETRY_HASH_NAME="craftax_llm_finish_retry_${SLURM_JOB_ID}"
COMPLETED_COUNTER_NAME="craftax_llm_finish_completed_${SLURM_JOB_ID}"

SUBMITTED_WORKER_JOBS=()
CLEANUP_DONE=0
MISSING_BASENAMES_FILE=""

cleanup() {
    if [[ "${CLEANUP_DONE}" -eq 1 ]]; then
        return
    fi
    CLEANUP_DONE=1

    if [[ "${#SUBMITTED_WORKER_JOBS[@]}" -gt 0 ]]; then
        echo "[cleanup] Cancelling worker arrays: ${SUBMITTED_WORKER_JOBS[*]}"
        scancel "${SUBMITTED_WORKER_JOBS[@]}" 2>/dev/null || true
    fi

    if [[ -x "${REDIS_DIR}/src/redis-cli" ]]; then
        "${REDIS_DIR}/src/redis-cli" -h "$(hostname)" -p "${REDIS_PORT}" SHUTDOWN NOSAVE >/dev/null 2>&1 || true
    fi

    if [[ -n "${MISSING_BASENAMES_FILE}" && -f "${MISSING_BASENAMES_FILE}" ]]; then
        rm -f "${MISSING_BASENAMES_FILE}" || true
    fi
}
trap cleanup EXIT INT TERM

count_active_workers() {
    local total=0
    local jid
    for jid in "${SUBMITTED_WORKER_JOBS[@]}"; do
        local c
        c=$(squeue -h -j "${jid}" -t R,PD 2>/dev/null | wc -l)
        total=$((total + c))
    done
    echo "${total}"
}

submit_workers() {
    local worker_count="$1"
    local export_vars="ALL,QUEUE_NAME=${QUEUE_NAME},RETRY_HASH_NAME=${RETRY_HASH_NAME},COMPLETED_COUNTER_NAME=${COMPLETED_COUNTER_NAME},REDIS_HOST=${REDIS_HOST},REDIS_PORT=${REDIS_PORT},RESULTS_DIR=${RESULTS_DIR},LOGS_DIR=${LOGS_DIR},VLLM_PORT_BASE=${VLLM_PORT_BASE}"
    local jid
    jid=$(sbatch --parsable --array=1-"${worker_count}" --export="${export_vars}" "${WORKER_SCRIPT}")
    SUBMITTED_WORKER_JOBS+=("${jid}")
    echo "Submitted worker array ${jid} with ${worker_count} tasks"
}

echo "=========================================="
echo "ONE-SHOT FINISH REMAINING LABELLING"
echo "=========================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Source: ${SOURCE_DIR}"
echo "Results: ${RESULTS_DIR}"
echo "Workers: ${NUM_WORKERS}"
echo "Monitor interval: ${MONITOR_INTERVAL}s"
echo "Max rounds: ${MAX_ROUNDS}"
echo "Max rescues/round: ${MAX_RESCUES_PER_ROUND}"
echo "Rescue cooldown: ${RESCUE_COOLDOWN_SEC}s"
echo "Queue name: ${QUEUE_NAME}"
echo "Redis: ${REDIS_HOST}:${REDIS_PORT}"
echo "vLLM port base: ${VLLM_PORT_BASE}"
echo "Worker script: ${WORKER_SCRIPT}"
echo "=========================================="

ENV_PATH_DEFAULT="/data/user_data/geney/.conda/envs/craftax_fast_llm"
ENV_BIN_DEFAULT="${ENV_PATH_DEFAULT}/bin"

if [[ -d "${ENV_BIN_DEFAULT}" ]]; then
    export PATH="${ENV_BIN_DEFAULT}:${PATH}"
fi

if ! command -v python >/dev/null 2>&1 || ! command -v vllm >/dev/null 2>&1; then
    if [[ -f "${HOME}/.bashrc" ]]; then
        # ~/.bashrc may reference unset vars; temporarily relax nounset.
        set +u
        source "${HOME}/.bashrc" >/dev/null 2>&1 || true
        set -u
    fi
    if command -v conda >/dev/null 2>&1; then
        conda activate "${ENV_PATH_DEFAULT}" >/dev/null 2>&1 || conda activate craftax_fast_llm >/dev/null 2>&1 || true
    fi
fi

if ! command -v python >/dev/null 2>&1 || ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: python/vllm runtime not available."
    echo "  python path: $(command -v python || echo missing)"
    echo "  vllm path: $(command -v vllm || echo missing)"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
mkdir -p "${BASELINES_DIR}/logs"
mkdir -p "${LOGS_DIR}"

echo "[1/4] Starting Redis on $(hostname):${REDIS_PORT}"
cd "${REDIS_DIR}"
./src/redis-cli -p "${REDIS_PORT}" SHUTDOWN NOSAVE 2>/dev/null || true
sleep 2
./src/redis-server --port "${REDIS_PORT}" --bind 0.0.0.0 --protected-mode no --daemonize yes
sleep 2
./src/redis-cli -p "${REDIS_PORT}" ping || { echo "ERROR: Redis failed to start"; exit 1; }

SOURCE_COUNT=$(find "${SOURCE_DIR}" -maxdepth 1 -name "${RESULTS_PATTERN}" | wc -l)
echo "[2/4] Finishing missing files (source_count=${SOURCE_COUNT})"

ROUND=1
while [[ "${ROUND}" -le "${MAX_ROUNDS}" ]]; do
    SOURCE_NAMES_FILE=$(mktemp)
    RESULT_NAMES_FILE=$(mktemp)
    MISSING_BASENAMES_FILE=$(mktemp)

    find "${SOURCE_DIR}" -maxdepth 1 -name "${RESULTS_PATTERN}" -printf '%f\n' | sort > "${SOURCE_NAMES_FILE}"
    find "${RESULTS_DIR}" -maxdepth 1 -name "${RESULTS_PATTERN}" -printf '%f\n' | sort > "${RESULT_NAMES_FILE}"
    comm -23 "${SOURCE_NAMES_FILE}" "${RESULT_NAMES_FILE}" > "${MISSING_BASENAMES_FILE}"

    COMPLETED_COUNT=$(wc -l < "${RESULT_NAMES_FILE}")
    MISSING_COUNT=$(wc -l < "${MISSING_BASENAMES_FILE}")

    echo "Round ${ROUND}/${MAX_ROUNDS}: completed=${COMPLETED_COUNT}/${SOURCE_COUNT}, missing=${MISSING_COUNT}"

    rm -f "${SOURCE_NAMES_FILE}" "${RESULT_NAMES_FILE}"

    if [[ "${MISSING_COUNT}" -eq 0 ]]; then
        break
    fi

    ./src/redis-cli -p "${REDIS_PORT}" DEL "${QUEUE_NAME}" >/dev/null
    ./src/redis-cli -p "${REDIS_PORT}" DEL "${RETRY_HASH_NAME}" >/dev/null
    ./src/redis-cli -p "${REDIS_PORT}" DEL "${COMPLETED_COUNTER_NAME}" >/dev/null

    while IFS= read -r base; do
        [[ -z "${base}" ]] && continue
        src_path="${SOURCE_DIR}/${base}"
        if [[ -f "${src_path}" ]]; then
            ./src/redis-cli -p "${REDIS_PORT}" LPUSH "${QUEUE_NAME}" "${src_path}" >/dev/null
        fi
    done < "${MISSING_BASENAMES_FILE}"

    INITIAL_QUEUE=$(./src/redis-cli -p "${REDIS_PORT}" LLEN "${QUEUE_NAME}")
    echo "Round ${ROUND}: initial queue size=${INITIAL_QUEUE}"
    if [[ "${INITIAL_QUEUE}" -eq 0 ]]; then
        echo "Round ${ROUND}: queue empty after filtering; re-checking filesystem next round."
        ROUND=$((ROUND + 1))
        continue
    fi

    echo "[3/4] Round ${ROUND}: submitting workers"
    INITIAL_WORKERS=$((INITIAL_QUEUE < NUM_WORKERS ? INITIAL_QUEUE : NUM_WORKERS))
    submit_workers "${INITIAL_WORKERS}"

    echo "[4/4] Round ${ROUND}: monitoring until queue drains and workers exit"
    RESCUES_THIS_ROUND=0
    LAST_RESCUE_EPOCH=0
    while true; do
        REMAINING=$(./src/redis-cli -p "${REDIS_PORT}" LLEN "${QUEUE_NAME}")
        ACTIVE=$(count_active_workers)
        DONE_NOW=$(find "${RESULTS_DIR}" -maxdepth 1 -name "${RESULTS_PATTERN}" | wc -l)
        echo "$(date '+%Y-%m-%d %H:%M:%S'): round=${ROUND} queue=${REMAINING} active_workers=${ACTIVE} completed=${DONE_NOW}/${SOURCE_COUNT}"

        if [[ "${REMAINING}" -eq 0 && "${ACTIVE}" -eq 0 ]]; then
            break
        fi

        # Rescue path: if queue still has jobs but all workers exited, submit one more batch.
        if [[ "${REMAINING}" -gt 0 && "${ACTIVE}" -eq 0 ]]; then
            NOW_EPOCH=$(date +%s)
            if [[ "${RESCUES_THIS_ROUND}" -ge "${MAX_RESCUES_PER_ROUND}" ]]; then
                echo "Round ${ROUND}: reached rescue limit (${MAX_RESCUES_PER_ROUND}). Rebuilding missing set next round."
                break
            fi
            if [[ "${LAST_RESCUE_EPOCH}" -gt 0 ]]; then
                AGE=$((NOW_EPOCH - LAST_RESCUE_EPOCH))
                if [[ "${AGE}" -lt "${RESCUE_COOLDOWN_SEC}" ]]; then
                    echo "Rescue cooldown active (${AGE}s < ${RESCUE_COOLDOWN_SEC}s); waiting."
                    sleep "${MONITOR_INTERVAL}"
                    continue
                fi
            fi
            RESCUE_WORKERS=$((REMAINING < NUM_WORKERS ? REMAINING : NUM_WORKERS))
            echo "Queue non-empty but no active workers. Submitting rescue batch of ${RESCUE_WORKERS} workers."
            submit_workers "${RESCUE_WORKERS}"
            RESCUES_THIS_ROUND=$((RESCUES_THIS_ROUND + 1))
            LAST_RESCUE_EPOCH="${NOW_EPOCH}"
        fi

        sleep "${MONITOR_INTERVAL}"
    done

    rm -f "${MISSING_BASENAMES_FILE}" || true
    MISSING_BASENAMES_FILE=""
    ROUND=$((ROUND + 1))
done

FINAL_DONE=$(find "${RESULTS_DIR}" -maxdepth 1 -name "${RESULTS_PATTERN}" | wc -l)
FINAL_MISSING=$((SOURCE_COUNT - FINAL_DONE))
echo "=========================================="
echo "ONE-SHOT FINISH COMPLETE"
echo "Final completed: ${FINAL_DONE}/${SOURCE_COUNT}"
echo "Final missing: ${FINAL_MISSING}"
echo "=========================================="
if [[ "${FINAL_MISSING}" -gt 0 ]]; then
    echo "ERROR: finish run exited with missing files still present."
    exit 1
fi
