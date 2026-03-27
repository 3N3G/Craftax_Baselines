#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_FILE="${ROOT_DIR}/scripts/sbatch/run_online_rl_hidden_jax.sbatch"
if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "Missing sbatch file: ${SBATCH_FILE}"
    exit 1
fi

ENVS="${ENVS:-128}"
TARGET_TIMESTEPS="${TARGET_TIMESTEPS:-300000000}"
SKIP_N="${SKIP_N:-5}"
LAYER="${LAYER:-24}"
TOKENS="${TOKENS:-1}"
NUM_STEPS="${NUM_STEPS:-64}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-10000000}"
POLICY_SAVE_DIR="${POLICY_SAVE_DIR:-/data/group_data/rl/geney/online_rl_hidden_models}"
HIDDEN_POOLING="${HIDDEN_POOLING:-last_token}"
HIDDEN_POOLING_K="${HIDDEN_POOLING_K:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SAVE_TRAJ_ONLINE="${SAVE_TRAJ_ONLINE:-0}"
TRAJ_SAVE_DIR="${TRAJ_SAVE_DIR:-}"
TRAJ_SAVE_EVERY_UPDATES="${TRAJ_SAVE_EVERY_UPDATES:-50}"
TRAJ_FREE_SPACE_MIN_GB="${TRAJ_FREE_SPACE_MIN_GB:-150}"
TRAJ_SCHEMA="${TRAJ_SCHEMA:-minimal_core}"
RUN_NAME="${RUN_NAME:-online-jax-128env-skip${SKIP_N}-target${TARGET_TIMESTEPS}_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${POLICY_SAVE_DIR}/${RUN_NAME}_resume}"
RESUME_FROM="${RESUME_FROM:-}"
JOB_NAME="${JOB_NAME:-orlhj_s${SKIP_N}_chain}"

echo "Submitting chain job"
echo "  job_name=${JOB_NAME}"
echo "  run_name=${RUN_NAME}"
echo "  target_timesteps=${TARGET_TIMESTEPS}"
echo "  skip_n=${SKIP_N} layer=${LAYER} tokens=${TOKENS}"
echo "  hidden_pooling=${HIDDEN_POOLING} k=${HIDDEN_POOLING_K} temp=${TEMPERATURE}"
echo "  checkpoint_dir=${CHECKPOINT_DIR}"
echo "  resume_from=${RESUME_FROM:-<none>}"
echo "  save_traj_online=${SAVE_TRAJ_ONLINE}"
if [[ "${SAVE_TRAJ_ONLINE}" == "1" ]]; then
    echo "  traj_save_dir=${TRAJ_SAVE_DIR:-<auto>}"
    echo "  traj_save_every_updates=${TRAJ_SAVE_EVERY_UPDATES}"
    echo "  traj_free_space_min_gb=${TRAJ_FREE_SPACE_MIN_GB}"
fi

EXPORTS=(
    "RUN_NAME=${RUN_NAME}"
    "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    "CHAIN_ENABLED=1"
    "CHAIN_TARGET_STEPS=${TARGET_TIMESTEPS}"
    "CHAIN_CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    "CHAIN_RUN_NAME=${RUN_NAME}"
    "HIDDEN_POOLING=${HIDDEN_POOLING}"
    "HIDDEN_POOLING_K=${HIDDEN_POOLING_K}"
    "TEMPERATURE=${TEMPERATURE}"
    "SAVE_TRAJ_ONLINE=${SAVE_TRAJ_ONLINE}"
    "TRAJ_SAVE_DIR=${TRAJ_SAVE_DIR}"
    "TRAJ_SAVE_EVERY_UPDATES=${TRAJ_SAVE_EVERY_UPDATES}"
    "TRAJ_FREE_SPACE_MIN_GB=${TRAJ_FREE_SPACE_MIN_GB}"
    "TRAJ_SCHEMA=${TRAJ_SCHEMA}"
)
if [[ -n "${RESUME_FROM}" ]]; then
    EXPORTS+=("RESUME_FROM=${RESUME_FROM}")
fi
EXPORT_STR="ALL,$(IFS=,; echo "${EXPORTS[*]}")"

sbatch \
    --job-name "${JOB_NAME}" \
    --export "${EXPORT_STR}" \
    "${SBATCH_FILE}" \
    "${ENVS}" \
    "${TARGET_TIMESTEPS}" \
    "${SKIP_N}" \
    "${LAYER}" \
    "${TOKENS}" \
    "${NUM_STEPS}" \
    "${CHECKPOINT_EVERY_STEPS}" \
    "${POLICY_SAVE_DIR}"
