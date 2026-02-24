#!/bin/bash
#SBATCH --job-name=awr_llm_offline
#SBATCH --output=/home/geney/Craftax_Baselines/logs/awr_llm_offline_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/awr_llm_offline_%j.err
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -eo pipefail

echo "Starting offline AWR-LLM job on $(hostname) at $(date)"

cd /home/geney/Craftax_Baselines
mkdir -p logs

ENV_PATH_DEFAULT="/data/user_data/geney/.conda/envs/craftax_fast_llm"
ENV_BIN_DEFAULT="${ENV_PATH_DEFAULT}/bin"

if [[ -d "${ENV_BIN_DEFAULT}" ]]; then
    export PATH="${ENV_BIN_DEFAULT}:${PATH}"
fi

if ! command -v python >/dev/null 2>&1; then
    if [[ -f "${HOME}/.bashrc" ]]; then
        source "${HOME}/.bashrc" >/dev/null 2>&1 || true
    fi
    if command -v conda >/dev/null 2>&1; then
        conda activate "${ENV_PATH_DEFAULT}" >/dev/null 2>&1 || conda activate craftax_fast_llm >/dev/null 2>&1 || true
    fi
fi

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python is not available after environment bootstrap."
    exit 1
fi

echo "Runtime python: $(command -v python)"
python -V || true

LOCAL_BASE="/scratch/${USER}"
if [[ ! -d "${LOCAL_BASE}" || ! -w "${LOCAL_BASE}" ]]; then
    LOCAL_BASE="/tmp/${USER}"
fi
mkdir -p "${LOCAL_BASE}"

RUN_ROOT="${LOCAL_BASE}/awr_llm_offline_${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}/wandb" "${RUN_ROOT}/wandb_cache" "${RUN_ROOT}/wandb_config"
export WANDB_DIR="${RUN_ROOT}/wandb"
export WANDB_CACHE_DIR="${RUN_ROOT}/wandb_cache"
export WANDB_CONFIG_DIR="${RUN_ROOT}/wandb_config"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=cpu
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-/data/group_data/rl/geney/vllm_craftax_labelled_results}"
DATA_GLOB="${DATA_GLOB:-trajectories_batch_*.npz}"
SAVE_DIR="${SAVE_DIR:-/data/group_data/rl/geney/checkpoints/awr_llm_augmented_live/${SLURM_JOB_ID}}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-3e-4}"
AWR_BETA="${AWR_BETA:-10.0}"
SEED="${SEED:-42}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
DATASET_WORKERS="${DATASET_WORKERS:-8}"
NUM_ENVS="${NUM_ENVS:-128}"
HIDDEN_MODE="${HIDDEN_MODE:-real}"
FUSION_MODE="${FUSION_MODE:-residual_gated}"
HIDDEN_GATE_INIT_LOGIT="${HIDDEN_GATE_INIT_LOGIT:--4.0}"
ADVANTAGE_MODE="${ADVANTAGE_MODE:-center}"
MIN_RTG_QUANTILE="${MIN_RTG_QUANTILE:-0.0}"
NO_WANDB="${NO_WANDB:-0}"
WANDB_NAME="${WANDB_NAME:-}"
MAX_DATASET_GB="${MAX_DATASET_GB:-80}"

CMD=(
    python offline_rl/awr_llm_augmented.py
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
    --hidden_mode "${HIDDEN_MODE}"
    --fusion_mode "${FUSION_MODE}"
    --hidden_gate_init_logit "${HIDDEN_GATE_INIT_LOGIT}"
    --advantage_mode "${ADVANTAGE_MODE}"
    --min_rtg_quantile "${MIN_RTG_QUANTILE}"
)

if [[ -n "${MAX_FILES:-}" ]]; then
    CMD+=(--max_files "${MAX_FILES}")
fi

if [[ "${REQUIRE_RETURNS:-0}" == "1" ]]; then
    CMD+=(--require-returns)
fi

if [[ "${DISABLE_AUTO_FILE_LIMIT:-0}" == "1" ]]; then
    CMD+=(--disable_auto_file_limit)
fi

if [[ "${NO_WANDB}" == "1" ]]; then
    CMD+=(--no_wandb)
fi

if [[ -n "${WANDB_NAME}" ]]; then
    CMD+=(--wandb_name "${WANDB_NAME}")
fi

echo "Launching training:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Offline AWR-LLM job finished at $(date)"
