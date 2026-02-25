#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_FILE="${ROOT_DIR}/scripts/sbatch/run_online_rl_hidden_jax.sbatch"
if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "Missing sbatch file: ${SBATCH_FILE}"
    exit 1
fi

ENVS="${ENVS:-128}"
TARGET_TIMESTEPS="${TARGET_TIMESTEPS:-100000000}"
SKIP_N="${SKIP_N:-5}"
LAYER="${LAYER:--1}"
TOKENS="${TOKENS:-1}"
NUM_STEPS="${NUM_STEPS:-64}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-10000000}"
POLICY_SAVE_DIR="${POLICY_SAVE_DIR:-/data/group_data/rl/geney/online_rl_hidden_models}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/group_data/rl/geney/online_rl_hidden_models/resume_state_skip5}"
RESUME_FROM="${RESUME_FROM:-}"
RUN_NAME="${RUN_NAME:-online-jax-128env-skip5-resume_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${RESUME_FROM}" ]]; then
    echo "ERROR: RESUME_FROM is required (checkpoint file or directory containing latest_resume.json)"
    exit 1
fi

echo "Submitting skip5 resume job"
echo "  resume_from: ${RESUME_FROM}"
echo "  run_name: ${RUN_NAME}"

sbatch \
    --job-name "orlhj_skip5_resume" \
    --export "ALL,RUN_NAME=${RUN_NAME},CHECKPOINT_DIR=${CHECKPOINT_DIR},RESUME_FROM=${RESUME_FROM}" \
    "${SBATCH_FILE}" \
    "${ENVS}" \
    "${TARGET_TIMESTEPS}" \
    "${SKIP_N}" \
    "${LAYER}" \
    "${TOKENS}" \
    "${NUM_STEPS}" \
    "${CHECKPOINT_EVERY_STEPS}" \
    "${POLICY_SAVE_DIR}"
