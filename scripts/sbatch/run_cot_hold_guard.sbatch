#!/bin/bash
#SBATCH --job-name=cot_hold_guard
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/cot_hold_guard_%j.out
#SBATCH --error=logs/cot_hold_guard_%j.err

set -u -o pipefail

cd ~/Craftax_Baselines
mkdir -p logs

POLL_SECONDS="${POLL_SECONDS:-120}"
JOB_NAME_REGEX="${JOB_NAME_REGEX:-^(cot_s|cot_promote_selector$)}"
GUARD_UNTIL_EPOCH="${GUARD_UNTIL_EPOCH:-0}"
CHAIN_ENABLED="${CHAIN_ENABLED:-1}"
CHAIN_GENERATION="${CHAIN_GENERATION:-0}"
SUCCESSOR_SUBMITTED=0

if ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 10 )); then
    echo "[hold-guard] invalid POLL_SECONDS='${POLL_SECONDS}', defaulting to 120"
    POLL_SECONDS=120
fi
if ! [[ "${GUARD_UNTIL_EPOCH}" =~ ^[0-9]+$ ]]; then
    echo "[hold-guard] invalid GUARD_UNTIL_EPOCH='${GUARD_UNTIL_EPOCH}', disabling end bound"
    GUARD_UNTIL_EPOCH=0
fi
if ! [[ "${CHAIN_GENERATION}" =~ ^[0-9]+$ ]]; then
    CHAIN_GENERATION=0
fi

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

scan_and_release_held() {
    local rows
    rows="$(squeue -u "${USER}" -t PD -h -o "%i|%j|%r" 2>/dev/null || true)"
    if [[ -z "${rows}" ]]; then
        echo "[hold-guard] $(timestamp) scan: no pending jobs"
        return 0
    fi

    local released=0
    local scanned=0
    while IFS='|' read -r job_id job_name reason; do
        [[ -z "${job_id}" ]] && continue
        scanned=$((scanned + 1))

        if [[ ! "${job_name}" =~ ${JOB_NAME_REGEX} ]]; then
            continue
        fi
        if [[ "${reason}" != "JobHeldUser" ]]; then
            continue
        fi

        echo "[hold-guard] $(timestamp) releasing held job id=${job_id} name=${job_name} reason=${reason}"
        if scontrol release "${job_id}" >/dev/null 2>&1; then
            released=$((released + 1))
        else
            echo "[hold-guard] $(timestamp) WARN failed to release job ${job_id}"
        fi
    done <<< "${rows}"

    echo "[hold-guard] $(timestamp) scan complete scanned=${scanned} released=${released}"
}

submit_successor() {
    if [[ "${CHAIN_ENABLED}" != "1" ]]; then
        return 0
    fi
    if [[ "${SUCCESSOR_SUBMITTED}" == "1" ]]; then
        return 0
    fi

    local now
    now="$(date +%s)"
    if (( GUARD_UNTIL_EPOCH > 0 && now >= GUARD_UNTIL_EPOCH )); then
        echo "[hold-guard] $(timestamp) guard horizon reached; no successor needed"
        return 0
    fi

    local next_generation=$((CHAIN_GENERATION + 1))
    local export_str
    export_str="$(printf "ALL,CHAIN_ENABLED=1,CHAIN_GENERATION=%s,GUARD_UNTIL_EPOCH=%s,POLL_SECONDS=%s,JOB_NAME_REGEX=%s" \
        "${next_generation}" \
        "${GUARD_UNTIL_EPOCH}" \
        "${POLL_SECONDS}" \
        "${JOB_NAME_REGEX}")"

    local submit_out
    submit_out="$(
        sbatch \
            --dependency="afterany:${SLURM_JOB_ID}" \
            --job-name "${SLURM_JOB_NAME}" \
            --export "${export_str}" \
            scripts/sbatch/run_cot_hold_guard.sbatch
    )"
    SUCCESSOR_SUBMITTED=1
    echo "[hold-guard] $(timestamp) submitted successor: ${submit_out}"
}

on_term() {
    echo "[hold-guard] $(timestamp) TERM/INT received; chaining next guard segment"
    submit_successor
    exit 0
}
trap on_term TERM INT

echo "[hold-guard] start job_id=${SLURM_JOB_ID} generation=${CHAIN_GENERATION} poll=${POLL_SECONDS}s regex='${JOB_NAME_REGEX}' until=${GUARD_UNTIL_EPOCH}"

while true; do
    local_now="$(date +%s)"
    if (( GUARD_UNTIL_EPOCH > 0 && local_now >= GUARD_UNTIL_EPOCH )); then
        echo "[hold-guard] $(timestamp) guard horizon reached; exiting"
        break
    fi

    scan_and_release_held
    sleep "${POLL_SECONDS}" &
    wait $!
done

submit_successor
echo "[hold-guard] $(timestamp) done"
