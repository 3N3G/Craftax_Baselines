#!/bin/bash
set -euo pipefail

# Launch two comparable offline AWR jobs:
# 1) real hidden states
# 2) zero hidden states (ablation baseline)
#
# Usage example:
#   TOTAL_STEPS=100000 MAX_FILES=512 ./scripts/sbatch/launch_offline_awr_pair.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_FILE="${ROOT_DIR}/scripts/sbatch/run_offline_awr_llm_augmented.sbatch"

if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "Missing sbatch file: ${SBATCH_FILE}"
    exit 1
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
COMMON_EXPORTS=(
    "ALL"
    "DATA_DIR=${DATA_DIR:-/data/group_data/rl/geney/vllm_craftax_labelled_results}"
    "DATA_GLOB=${DATA_GLOB:-trajectories_batch_*.npz}"
    "SAVE_DIR_BASE=${SAVE_DIR_BASE:-/data/group_data/rl/geney/checkpoints/awr_llm_augmented_live}"
    "TOTAL_STEPS=${TOTAL_STEPS:-100000}"
    "BATCH_SIZE=${BATCH_SIZE:-256}"
    "LR=${LR:-3e-4}"
    "AWR_BETA=${AWR_BETA:-10.0}"
    "SEED=${SEED:-42}"
    "SAVE_FREQ=${SAVE_FREQ:-25000}"
    "DATASET_WORKERS=${DATASET_WORKERS:-8}"
    "NUM_ENVS=${NUM_ENVS:-128}"
    "MAX_DATASET_GB=${MAX_DATASET_GB:-24}"
    "FUSION_MODE=${FUSION_MODE:-concat_raw}"
    "HIDDEN_GATE_INIT_LOGIT=${HIDDEN_GATE_INIT_LOGIT:--4.0}"
    "NO_WANDB=${NO_WANDB:-0}"
)

if [[ -n "${MAX_FILES:-}" ]]; then
    COMMON_EXPORTS+=("MAX_FILES=${MAX_FILES}")
fi

if [[ "${REQUIRE_RETURNS:-0}" == "1" ]]; then
    COMMON_EXPORTS+=("REQUIRE_RETURNS=1")
fi

if [[ "${DISABLE_AUTO_FILE_LIMIT:-0}" == "1" ]]; then
    COMMON_EXPORTS+=("DISABLE_AUTO_FILE_LIMIT=1")
fi

submit_job() {
    local mode="$1"
    local save_dir="${SAVE_DIR_BASE:-/data/group_data/rl/geney/checkpoints/awr_llm_augmented_live}/${RUN_TAG}_${mode}"
    local wandb_name="awr-llm-${RUN_TAG}-${mode}"
    local exports
    exports="$(IFS=,; echo "${COMMON_EXPORTS[*]},SAVE_DIR=${save_dir},HIDDEN_MODE=${mode},WANDB_NAME=${wandb_name}")"
    sbatch \
        --job-name="awr_${mode}" \
        --export="${exports}" \
        "${SBATCH_FILE}" \
        | awk '{print $4}'
}

REAL_JOB_ID="$(submit_job real)"
ZERO_JOB_ID="$(submit_job zero)"

echo "Submitted paired runs:"
echo "  real hidden states: ${REAL_JOB_ID}"
echo "  zero hidden states: ${ZERO_JOB_ID}"
echo "Run tag: ${RUN_TAG}"
