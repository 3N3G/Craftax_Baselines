#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_SBATCH="${ROOT_DIR}/scripts/sbatch/run_eval_policy_wave_id.sbatch"

if [[ ! -f "${RUN_SBATCH}" ]]; then
    echo "Missing sbatch runner: ${RUN_SBATCH}"
    exit 1
fi

RUN_TAG="${RUN_TAG:-offline9_shards_$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${MANIFEST:-configs/eval/policy_wave_v2_offline9_only.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/policy_wave_v2_offline9_shards/${RUN_TAG}}"
TRACKS="${TRACKS:-id}"
SEEDS="${SEEDS:-42}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EPISODES="${NUM_EPISODES:-128}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-80000}"
INCLUDE_SLICES="${INCLUDE_SLICES:-0}"
START_VLLM="${START_VLLM:-1}"
NO_WANDB="${NO_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-craftax_symbolic_evals}"
WANDB_ENTITY="${WANDB_ENTITY:-iris-sobolmark}"

cd "${ROOT_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Manifest not found: ${MANIFEST}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/shards"

mapfile -t POLICY_IDS < <(
    awk '/^[[:space:]]*-[[:space:]]*id:[[:space:]]*/ {print $3}' "${MANIFEST}"
)

if [[ ${#POLICY_IDS[@]} -eq 0 ]]; then
    echo "No policy ids found in manifest: ${MANIFEST}"
    exit 1
fi

echo "========================================"
echo "Submitting offline-9 shard evals"
echo "manifest: ${MANIFEST}"
echo "output_dir: ${OUTPUT_DIR}"
echo "tracks: ${TRACKS}"
echo "policy_count: ${#POLICY_IDS[@]}"
echo "========================================"

submitted=0
for pid in "${POLICY_IDS[@]}"; do
    safe_pid="$(echo "${pid}" | tr '/:' '__')"
    summary_rel="shards/${safe_pid}.json"
    job_name="eval9_${safe_pid}"

    jid="$(
        sbatch --parsable \
            --job-name "${job_name}" \
            --export "ALL,MANIFEST=${MANIFEST},TRACKS=${TRACKS},OUTPUT_DIR=${OUTPUT_DIR},SEEDS=${SEEDS},NUM_ENVS=${NUM_ENVS},NUM_EPISODES=${NUM_EPISODES},MAX_ENV_STEPS=${MAX_ENV_STEPS},INCLUDE_SLICES=${INCLUDE_SLICES},START_VLLM=${START_VLLM},NO_WANDB=${NO_WANDB},WANDB_PROJECT=${WANDB_PROJECT},WANDB_ENTITY=${WANDB_ENTITY},POLICY_IDS=${pid},SUMMARY_PATH=${summary_rel}" \
            "${RUN_SBATCH}"
    )"

    echo "  ${pid} -> ${jid} (${summary_rel})"
    submitted=$((submitted + 1))
    if [[ -n "${JOB_IDS_OUT:-}" ]]; then
        echo "${jid}" >> "${JOB_IDS_OUT}"
    fi
done

echo "Submitted ${submitted} shard jobs."
