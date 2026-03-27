#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PPO_SBATCH="${ROOT_DIR}/scripts/sbatch/run_ppo_symbolic_policy.sbatch"
OFFLINE_GRID_LAUNCHER="${ROOT_DIR}/scripts/sbatch/launch_offline_awr_skip_grid.sh"

if [[ ! -f "${PPO_SBATCH}" ]]; then
    echo "Missing PPO sbatch: ${PPO_SBATCH}"
    exit 1
fi
if [[ ! -f "${OFFLINE_GRID_LAUNCHER}" ]]; then
    echo "Missing offline grid launcher: ${OFFLINE_GRID_LAUNCHER}"
    exit 1
fi

RUN_TAG="${RUN_TAG:-wavev2_$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${MANIFEST:-configs/eval/policy_wave_v2.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/policy_wave_v2/${RUN_TAG}}"
SUBMIT_PPO="${SUBMIT_PPO:-1}"
SUBMIT_OFFLINE_GRID="${SUBMIT_OFFLINE_GRID:-1}"
SUBMIT_EVALS="${SUBMIT_EVALS:-1}"

# Existing policy-production jobs already queued/running can be injected here.
# Example: EXISTING_POLICY_JOB_IDS="6436639,6436640,6436641,online_skip25_jobid,online_skip5_jobid"
EXISTING_POLICY_JOB_IDS="${EXISTING_POLICY_JOB_IDS:-}"

echo "========================================"
echo "Policy Wave v2 launcher"
echo "run_tag: ${RUN_TAG}"
echo "manifest: ${MANIFEST}"
echo "output_dir: ${OUTPUT_DIR}"
echo "========================================"

if command -v squeue >/dev/null 2>&1; then
    echo "Current queue snapshot for online skip jobs (monitor only, no resubmit):"
    squeue -u "${USER}" -h -o "%i %j %t %M %R" | grep -E "skip25|skip5" || true
fi

submitted_ids=()

if [[ "${SUBMIT_PPO}" == "1" ]]; then
    echo "Submitting PPO symbolic baseline with checkpoint slicing..."
    ppo_id=$(sbatch \
        --job-name "ppo_symbolic_wavev2" \
        --export "ALL,RUN_NAME=ppo_symbolic_${RUN_TAG}" \
        "${PPO_SBATCH}" | awk '{print $4}')
    echo "  ppo job: ${ppo_id}"
    submitted_ids+=("${ppo_id}")
else
    echo "SUBMIT_PPO=0 -> skipping PPO submission"
fi

if [[ "${SUBMIT_OFFLINE_GRID}" == "1" ]]; then
    echo "Submitting offline 9-grid (queue/completion-aware)..."
    JOB_IDS_FILE="$(mktemp)"
    (
        cd "${ROOT_DIR}"
        JOB_IDS_OUT="${JOB_IDS_FILE}" \
        RUN_TAG="${RUN_TAG}" \
        bash "${OFFLINE_GRID_LAUNCHER}"
    )
    if [[ -s "${JOB_IDS_FILE}" ]]; then
        while IFS= read -r jid; do
            [[ -n "${jid}" ]] && submitted_ids+=("${jid}")
        done < "${JOB_IDS_FILE}"
    fi
    rm -f "${JOB_IDS_FILE}"
else
    echo "SUBMIT_OFFLINE_GRID=0 -> skipping offline grid submission"
fi

dep_ids=()
for jid in "${submitted_ids[@]}"; do
    if [[ "${jid}" =~ ^[0-9]+$ ]]; then
        dep_ids+=("${jid}")
    fi
done
if [[ -n "${EXISTING_POLICY_JOB_IDS}" ]]; then
    IFS=',' read -r -a extra_ids <<< "${EXISTING_POLICY_JOB_IDS}"
    for jid in "${extra_ids[@]}"; do
        jid="$(echo "${jid}" | xargs)"
        if [[ "${jid}" =~ ^[0-9]+$ ]]; then
            dep_ids+=("${jid}")
        fi
    done
fi

# Deduplicate dependency ids.
if [[ ${#dep_ids[@]} -gt 0 ]]; then
    dep_csv="$(printf "%s\n" "${dep_ids[@]}" | awk '!seen[$0]++' | paste -sd, -)"
    dep_arg=(--dependency "afterok:${dep_csv}")
    echo "Policy dependency ids: ${dep_csv}"
else
    dep_arg=()
    echo "No policy dependency ids resolved; evals will submit immediately."
fi

if [[ "${SUBMIT_EVALS}" == "1" ]]; then
    echo "Submitting evaluation tracks on rl partition..."

    common_export="ALL,MANIFEST=${MANIFEST},OUTPUT_DIR=${OUTPUT_DIR},RUN_TAG=${RUN_TAG}"
    eval_ids=()

    id_job=$(sbatch "${dep_arg[@]}" \
        --job-name "wavev2_id" \
        --export "${common_export},TRACKS=id,START_VLLM=1" \
        "${ROOT_DIR}/scripts/sbatch/run_eval_policy_wave_id.sbatch" | awk '{print $4}')
    echo "  id job: ${id_job}"
    eval_ids+=("${id_job}")

    value_job=$(sbatch "${dep_arg[@]}" \
        --job-name "wavev2_value" \
        --export "${common_export},TRACKS=value,START_VLLM=0" \
        "${ROOT_DIR}/scripts/sbatch/run_eval_policy_wave_value.sbatch" | awk '{print $4}')
    echo "  value job: ${value_job}"
    eval_ids+=("${value_job}")

    ood_job=$(sbatch "${dep_arg[@]}" \
        --job-name "wavev2_ood" \
        --export "${common_export},TRACKS=ood,START_VLLM=1" \
        "${ROOT_DIR}/scripts/sbatch/run_eval_policy_wave_ood.sbatch" | awk '{print $4}')
    echo "  ood job: ${ood_job}"
    eval_ids+=("${ood_job}")

    llm_job=$(sbatch "${dep_arg[@]}" \
        --job-name "wavev2_llm" \
        --export "${common_export},TRACKS=gameplay_llm,START_VLLM=1" \
        "${ROOT_DIR}/scripts/sbatch/run_eval_policy_wave_gameplay_llm.sbatch" | awk '{print $4}')
    echo "  gameplay_llm job: ${llm_job}"
    eval_ids+=("${llm_job}")

    eval_dep_csv="$(printf "%s\n" "${eval_ids[@]}" | awk '!seen[$0]++' | paste -sd, -)"
    summary_json="${OUTPUT_DIR}/policy_wave_v2_summary.json"
    report_md="${ROOT_DIR}/analysis/reports/policy_wave_v2_report.md"
    report_job=$(sbatch \
        --dependency "afterok:${eval_dep_csv}" \
        --job-name "wavev2_report" \
        --export "ALL,SUMMARY_JSON=${summary_json},OUTPUT_MD=${report_md}" \
        "${ROOT_DIR}/scripts/sbatch/run_policy_wave_report.sbatch" | awk '{print $4}')
    echo "  report job: ${report_job}"

    echo "Evaluation jobs submitted."
else
    echo "SUBMIT_EVALS=0 -> skipping evaluation submissions"
fi
