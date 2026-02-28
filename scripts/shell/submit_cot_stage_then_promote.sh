#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

POLICY_SAVE_DIR="${POLICY_SAVE_DIR:-/data/group_data/rl/geney/online_rl_hidden_models}"
STAGE_TARGET="${STAGE_TARGET:-60000000}"
PROMOTE_TARGET="${PROMOTE_TARGET:-300000000}"
ENVS="${ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-64}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-10000000}"
LAYER="${LAYER:-24}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SAVE_TRAJ_ONLINE="${SAVE_TRAJ_ONLINE:-0}"
TRAJ_SAVE_EVERY_UPDATES="${TRAJ_SAVE_EVERY_UPDATES:-50}"
TRAJ_FREE_SPACE_MIN_GB="${TRAJ_FREE_SPACE_MIN_GB:-150}"
TRAJ_SCHEMA="${TRAJ_SCHEMA:-minimal_core}"
ENABLE_COT_HOLD_GUARD="${ENABLE_COT_HOLD_GUARD:-1}"
COT_HOLD_GUARD_DAYS="${COT_HOLD_GUARD_DAYS:-8}"
COT_HOLD_GUARD_POLL_SECONDS="${COT_HOLD_GUARD_POLL_SECONDS:-120}"

TS="$(date +%Y%m%d_%H%M%S)"
MANIFEST_DIR="analysis/reports"
mkdir -p "${MANIFEST_DIR}"
MANIFEST_PATH="${MANIFEST_DIR}/cot_stage_manifest_${TS}.json"

declare -a JOB_IDS
declare -a ENTRIES

submit_one() {
    local skip="$1"
    local tokens="$2"
    local pooling="$3"
    local pooling_k="$4"
    local run_name="online-cot-s${skip}-tok${tokens}-${pooling}-stage60m-${TS}"
    local checkpoint_dir="${POLICY_SAVE_DIR}/${run_name}_resume"
    local job_name="cot_s${skip}_t${tokens}_${pooling}"
    local traj_dir=""
    if [[ "${SAVE_TRAJ_ONLINE}" == "1" ]]; then
        traj_dir="${POLICY_SAVE_DIR}/online_traj/${run_name}"
    fi

    local submit_out
    submit_out=$(
        ENVS="${ENVS}" \
        TARGET_TIMESTEPS="${STAGE_TARGET}" \
        SKIP_N="${skip}" \
        LAYER="${LAYER}" \
        TOKENS="${tokens}" \
        NUM_STEPS="${NUM_STEPS}" \
        CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS}" \
        POLICY_SAVE_DIR="${POLICY_SAVE_DIR}" \
        HIDDEN_POOLING="${pooling}" \
        HIDDEN_POOLING_K="${pooling_k}" \
        TEMPERATURE="${TEMPERATURE}" \
        SAVE_TRAJ_ONLINE="${SAVE_TRAJ_ONLINE}" \
        TRAJ_SAVE_DIR="${traj_dir}" \
        TRAJ_SAVE_EVERY_UPDATES="${TRAJ_SAVE_EVERY_UPDATES}" \
        TRAJ_FREE_SPACE_MIN_GB="${TRAJ_FREE_SPACE_MIN_GB}" \
        TRAJ_SCHEMA="${TRAJ_SCHEMA}" \
        RUN_NAME="${run_name}" \
        CHECKPOINT_DIR="${checkpoint_dir}" \
        JOB_NAME="${job_name}" \
        bash scripts/shell/submit_online_rl_hidden_chain.sh
    )
    local job_id
    job_id="$(echo "${submit_out}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
    if [[ -z "${job_id}" ]]; then
        echo "ERROR: failed to parse job id from submit output"
        echo "${submit_out}"
        exit 1
    fi
    JOB_IDS+=("${job_id}")
    ENTRIES+=("{\"job_id\":${job_id},\"job_name\":\"${job_name}\",\"run_name\":\"${run_name}\",\"checkpoint_dir\":\"${checkpoint_dir}\",\"policy_save_dir\":\"${POLICY_SAVE_DIR}\",\"envs\":${ENVS},\"num_steps\":${NUM_STEPS},\"checkpoint_every_steps\":${CHECKPOINT_EVERY_STEPS},\"skip_n\":${skip},\"layer\":${LAYER},\"tokens\":${tokens},\"hidden_pooling\":\"${pooling}\",\"hidden_pooling_k\":${pooling_k},\"temperature\":${TEMPERATURE},\"save_traj_online\":${SAVE_TRAJ_ONLINE},\"traj_save_dir\":\"${traj_dir}\",\"traj_save_every_updates\":${TRAJ_SAVE_EVERY_UPDATES},\"traj_free_space_min_gb\":${TRAJ_FREE_SPACE_MIN_GB},\"traj_schema\":\"${TRAJ_SCHEMA}\"}")
    echo "Submitted ${job_name}: job_id=${job_id}"
}

for skip in 5 25; do
    for tokens in 64 256; do
        submit_one "${skip}" "${tokens}" "last_token" 1
        submit_one "${skip}" "${tokens}" "mean_last_k" 8
    done
done

{
    echo "["
    for i in "${!ENTRIES[@]}"; do
        if [[ $i -gt 0 ]]; then
            echo ","
        fi
        echo -n "${ENTRIES[$i]}"
    done
    echo
    echo "]"
} > "${MANIFEST_PATH}"
echo "Wrote stage manifest: ${MANIFEST_PATH}"

DEP_STR="$(IFS=:; echo "${JOB_IDS[*]}")"
SELECTOR_OUT=$(
    sbatch \
        --dependency="afterany:${DEP_STR}" \
        scripts/sbatch/run_cot_promotion_selector.sbatch \
        "${MANIFEST_PATH}" \
        2 \
        "${PROMOTE_TARGET}"
)
echo "Submitted selector: ${SELECTOR_OUT}"

if [[ "${ENABLE_COT_HOLD_GUARD}" == "1" ]]; then
    echo "Ensuring CoT held-job guard is active..."
    DAYS="${COT_HOLD_GUARD_DAYS}" \
    POLL_SECONDS="${COT_HOLD_GUARD_POLL_SECONDS}" \
    JOB_NAME_REGEX="^(cot_s|cot_promote_selector$)" \
    JOB_NAME="cot_hold_guard" \
    PARTITION="cpu" \
    bash scripts/shell/submit_cot_hold_guard.sh
fi
