#!/bin/bash
#SBATCH --job-name=awr_unaug
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/awr_unaugmented_%j.out
#SBATCH --error=logs/awr_unaugmented_%j.err

# Unaugmented offline AWR training (same arch as PPO ActorCritic, no hidden states).

set -Eeo pipefail

# Source bashrc defensively because some profile scripts can return non-zero.
set +e
set +u
set +o pipefail
source ~/.bashrc
BASHRC_RC=$?
set -e
set -u
set -o pipefail
if [[ $BASHRC_RC -ne 0 ]]; then
    echo "WARNING: source ~/.bashrc exited with code $BASHRC_RC; continuing"
fi

mkdir -p logs
cd ~/Craftax_Baselines
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Mirror runtime logs to a non-repo path.
RUNTIME_LOG_ROOT="${RUNTIME_LOG_ROOT:-/data/user_data/${USER}/craftax_job_logs}"
mkdir -p "${RUNTIME_LOG_ROOT}" >/dev/null 2>&1 || true
RUNTIME_STDOUT="${RUNTIME_LOG_ROOT}/awr_unaugmented_${SLURM_JOB_ID}.runtime.out"
RUNTIME_STDERR="${RUNTIME_LOG_ROOT}/awr_unaugmented_${SLURM_JOB_ID}.runtime.err"
exec > >(tee -a "${RUNTIME_STDOUT}")
exec 2> >(tee -a "${RUNTIME_STDERR}" >&2)
echo "Runtime log mirror: ${RUNTIME_STDOUT}"
echo "Runtime error mirror: ${RUNTIME_STDERR}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found after sourcing bashrc"
    exit 1
fi

activate_conda_for() {
    local purpose="$1"
    shift
    local env_name
    for env_name in "$@"; do
        if conda activate "${env_name}" >/dev/null 2>&1; then
            echo "Activated conda env for ${purpose}: ${env_name}"
            return 0
        fi
    done
    echo "ERROR: failed to activate conda env for ${purpose}"
    echo "Tried: $*"
    conda env list || true
    return 1
}

on_error() {
    local rc=$?
    echo "FATAL: run_offline_awr_unaugmented.sbatch failed (exit=${rc}) at line ${BASH_LINENO[0]} while running: ${BASH_COMMAND}"
    exit "${rc}"
}
trap on_error ERR

# Activate training env
echo "Activating training environment..."
activate_conda_for "training" \
    /data/user_data/geney/.conda/envs/imaug \
    imaug \
    /data/user_data/geney/.conda/envs/craftax \
    craftax \
    base

# Verify imports
python -c 'import torch; print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")'

# Parameters
DATA_DIR="${DATA_DIR:-/data/group_data/rl/geney/vllm_craftax_labelled_results}"
DATA_GLOB="${DATA_GLOB:-trajectories_batch_*.npz}"
DEFAULT_DATA_DIR="/data/group_data/rl/geney/vllm_craftax_labelled_results"
SAVE_DIR="${SAVE_DIR:-/data/group_data/rl/geney/checkpoints/awr_unaugmented/${SLURM_JOB_ID}}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-3e-4}"
AWR_BETA="${AWR_BETA:-10.0}"
SEED="${SEED:-42}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
MAX_FILES="${MAX_FILES:-}"
DATASET_WORKERS="${DATASET_WORKERS:-8}"
NUM_ENVS="${NUM_ENVS:-128}"
ADVANTAGE_MODE="${ADVANTAGE_MODE:-center}"
MIN_RTG_QUANTILE="${MIN_RTG_QUANTILE:-0.0}"
NO_WANDB="${NO_WANDB:-0}"
WANDB_NAME="${WANDB_NAME:-}"
MAX_DATASET_GB="${MAX_DATASET_GB:-80}"
DATASET_SHARD_STEPS="${DATASET_SHARD_STEPS:-0}"

if ! compgen -G "${DATA_DIR}/${DATA_GLOB}" > /dev/null; then
    echo "WARNING: no files matched DATA_DIR='${DATA_DIR}' DATA_GLOB='${DATA_GLOB}'"
    if [[ "${DATA_DIR}" != "${DEFAULT_DATA_DIR}" ]] && compgen -G "${DEFAULT_DATA_DIR}/${DATA_GLOB}" > /dev/null; then
        echo "Falling back to default DATA_DIR='${DEFAULT_DATA_DIR}'"
        DATA_DIR="${DEFAULT_DATA_DIR}"
    else
        echo "ERROR: no dataset files found for '${DATA_DIR}/${DATA_GLOB}'"
        echo "If this is intentional, pass a valid DATA_DIR and DATA_GLOB."
        exit 1
    fi
fi

echo "========================================"
echo "Unaugmented AWR Offline RL"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --list-gpus | head -1)"
echo "Git commit: ${GIT_COMMIT}"
echo "========================================"
echo "Data dir: ${DATA_DIR}"
echo "Save dir: ${SAVE_DIR}"
echo "Total steps: ${TOTAL_STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "AWR beta: ${AWR_BETA}"
echo "Seed: ${SEED}"
echo ""

CMD=(
    python offline_rl/awr_unaugmented.py
    --data_dir "${DATA_DIR}"
    --data_glob "${DATA_GLOB}"
    --save_dir "${SAVE_DIR}"
    --total_steps "${TOTAL_STEPS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --awr_beta "${AWR_BETA}"
    --seed "${SEED}"
    --save_freq "${SAVE_FREQ}"
    --dataset_workers "${DATASET_WORKERS}"
    --num_envs "${NUM_ENVS}"
    --max_dataset_gb "${MAX_DATASET_GB}"
    --dataset_shard_steps "${DATASET_SHARD_STEPS}"
    --advantage_mode "${ADVANTAGE_MODE}"
    --min_rtg_quantile "${MIN_RTG_QUANTILE}"
)

if [[ -n "${MAX_FILES}" ]]; then
    CMD+=(--max_files "${MAX_FILES}")
fi

if [[ -n "${WANDB_NAME}" ]]; then
    CMD+=(--wandb_name "${WANDB_NAME}")
fi

if [[ "${NO_WANDB}" == "1" ]]; then
    CMD+=(--no_wandb)
fi

if [[ "${DISABLE_AUTO_FILE_LIMIT:-0}" == "1" ]]; then
    CMD+=(--disable_auto_file_limit)
fi

if [[ "${DISABLE_DATASET_SHARD_ROTATION:-0}" == "1" ]]; then
    CMD+=(--disable_dataset_shard_rotation)
fi

printf '  %q' "${CMD[@]}"
echo
set +e
"${CMD[@]}"
TRAIN_RC=$?
set -e
if [[ ${TRAIN_RC} -ne 0 ]]; then
    echo "ERROR: training process exited with code ${TRAIN_RC}"
    exit "${TRAIN_RC}"
fi

echo ""
echo "Done! Date: $(date)"
