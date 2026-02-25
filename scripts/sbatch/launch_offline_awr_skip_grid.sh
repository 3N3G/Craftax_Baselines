#!/bin/bash
set -euo pipefail

# Launch offline AWR-LLM grid over fusion mode x hidden skip cadence.
# Default matrix:
#   fusion in {concat_raw, gated_proj, residual_gated}
#   hidden_skip_n in {1, 5, 25}
#
# Existing jobs/checkpoints are skipped by default.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_FILE="${ROOT_DIR}/scripts/sbatch/run_offline_awr_llm_augmented.sbatch"

if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "Missing sbatch file: ${SBATCH_FILE}"
    exit 1
fi

parse_csv() {
    local csv="$1"
    local out_name="$2"
    local tmp=()
    IFS=',' read -r -a tmp <<< "${csv}"
    local cleaned=()
    local v
    for v in "${tmp[@]}"; do
        v="$(echo "${v}" | xargs)"
        if [[ -n "${v}" ]]; then
            cleaned+=("${v}")
        fi
    done
    eval "${out_name}=()"
    if [[ ${#cleaned[@]} -gt 0 ]]; then
        eval "${out_name}=(\"\${cleaned[@]}\")"
    fi
}

FUSIONS_CSV="${FUSIONS:-concat_raw,gated_proj,residual_gated}"
SKIPS_CSV="${SKIPS:-1,5,25}"
parse_csv "${FUSIONS_CSV}" FUSIONS
parse_csv "${SKIPS_CSV}" SKIPS

RUN_TAG="${RUN_TAG:-offline_grid_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR_BASE="${SAVE_DIR_BASE:-/data/group_data/rl/geney/checkpoints/awr_llm_augmented_live}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SKIP_QUEUED="${SKIP_QUEUED:-1}"
DRY_RUN="${DRY_RUN:-0}"

common_exports=(
    "ALL"
    "DATA_DIR=${DATA_DIR:-/data/group_data/rl/geney/vllm_craftax_labelled_results}"
    "DATA_GLOB=${DATA_GLOB:-trajectories_batch_*.npz}"
    "TOTAL_STEPS=${TOTAL_STEPS:-100000}"
    "BATCH_SIZE=${BATCH_SIZE:-256}"
    "LR=${LR:-3e-4}"
    "AWR_BETA=${AWR_BETA:-10.0}"
    "SEED=${SEED:-42}"
    "SAVE_FREQ=${SAVE_FREQ:-25000}"
    "DATASET_WORKERS=${DATASET_WORKERS:-8}"
    "NUM_ENVS=${NUM_ENVS:-128}"
    "MAX_DATASET_GB=${MAX_DATASET_GB:-24}"
    "HIDDEN_MODE=real"
    "NO_WANDB=${NO_WANDB:-0}"
)

if [[ -n "${MAX_FILES:-}" ]]; then
    common_exports+=("MAX_FILES=${MAX_FILES}")
fi
if [[ "${REQUIRE_RETURNS:-0}" == "1" ]]; then
    common_exports+=("REQUIRE_RETURNS=1")
fi
if [[ "${DISABLE_AUTO_FILE_LIMIT:-0}" == "1" ]]; then
    common_exports+=("DISABLE_AUTO_FILE_LIMIT=1")
fi
if [[ "${HIDDEN_SKIP_RESET_ON_DONE:-0}" == "1" ]]; then
    common_exports+=("HIDDEN_SKIP_RESET_ON_DONE=1")
fi

is_queued() {
    local job_name="$1"
    if ! command -v squeue >/dev/null 2>&1; then
        return 1
    fi
    squeue -h -u "${USER}" -n "${job_name}" | grep -q .
}

submitted=0
skipped_completed=0
skipped_queued=0
submitted_ids=()

for fusion in "${FUSIONS[@]}"; do
    for skip in "${SKIPS[@]}"; do
        if ! [[ "${skip}" =~ ^[0-9]+$ ]] || [[ "${skip}" -lt 1 ]]; then
            echo "Invalid skip value: ${skip}"
            exit 1
        fi

        run_key="${RUN_TAG}_${fusion}_skip${skip}_real"
        save_dir="${SAVE_DIR_BASE}/${run_key}"
        wandb_name="${run_key}"
        job_name="awr_${fusion}_s${skip}"

        if [[ "${SKIP_COMPLETED}" == "1" && -f "${save_dir}/awr_llm_final.pth" ]]; then
            echo "[skip-completed] ${run_key} (${save_dir}/awr_llm_final.pth exists)"
            skipped_completed=$((skipped_completed + 1))
            continue
        fi

        if [[ "${SKIP_QUEUED}" == "1" ]] && is_queued "${job_name}"; then
            echo "[skip-queued] ${run_key} (job name ${job_name} already in queue)"
            skipped_queued=$((skipped_queued + 1))
            continue
        fi

        exports="$(IFS=,; echo "${common_exports[*]},SAVE_DIR=${save_dir},WANDB_NAME=${wandb_name},FUSION_MODE=${fusion},HIDDEN_SKIP_N=${skip}")"
        cmd=(
            sbatch
            --job-name "${job_name}"
            --export "${exports}"
            "${SBATCH_FILE}"
        )

        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "[dry-run] ${cmd[*]}"
            continue
        fi

        job_id="$(${cmd[@]} | awk '{print $4}')"
        echo "[submitted] ${run_key} -> ${job_id}"
        submitted=$((submitted + 1))
        submitted_ids+=("${job_id}")
    done

done

echo ""
echo "Grid submission summary"
echo "  run_tag: ${RUN_TAG}"
echo "  submitted: ${submitted}"
echo "  skipped_completed: ${skipped_completed}"
echo "  skipped_queued: ${skipped_queued}"

if [[ -n "${JOB_IDS_OUT:-}" ]]; then
    mkdir -p "$(dirname "${JOB_IDS_OUT}")"
    printf "%s\n" "${submitted_ids[@]}" > "${JOB_IDS_OUT}"
    echo "  job_ids_out: ${JOB_IDS_OUT}"
fi
