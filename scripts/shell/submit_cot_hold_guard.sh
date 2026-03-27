#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_FILE="${ROOT_DIR}/scripts/sbatch/run_cot_hold_guard.sbatch"
if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "Missing sbatch file: ${SBATCH_FILE}"
    exit 1
fi

DAYS="${DAYS:-8}"
POLL_SECONDS="${POLL_SECONDS:-120}"
JOB_NAME_REGEX="${JOB_NAME_REGEX:-^(cot_s|cot_promote_selector$)}"
PARTITION="${PARTITION:-cpu}"
JOB_NAME="${JOB_NAME:-cot_hold_guard}"

if ! [[ "${DAYS}" =~ ^[0-9]+$ ]] || (( DAYS < 1 )); then
    echo "ERROR: DAYS must be a positive integer (got '${DAYS}')"
    exit 1
fi
if ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 10 )); then
    echo "ERROR: POLL_SECONDS must be an integer >= 10 (got '${POLL_SECONDS}')"
    exit 1
fi

EXISTING="$(squeue -u "${USER}" -h -n "${JOB_NAME}" -o "%i|%T|%j" || true)"
if [[ -n "${EXISTING}" ]]; then
    echo "Guard job '${JOB_NAME}' already exists:"
    echo "${EXISTING}"
    exit 0
fi

UNTIL_EPOCH="$(( $(date +%s) + DAYS * 86400 ))"
EXPORT_STR="$(printf "ALL,CHAIN_ENABLED=1,CHAIN_GENERATION=0,GUARD_UNTIL_EPOCH=%s,POLL_SECONDS=%s,JOB_NAME_REGEX=%s" \
    "${UNTIL_EPOCH}" \
    "${POLL_SECONDS}" \
    "${JOB_NAME_REGEX}")"

echo "Submitting CoT hold guard"
echo "  partition=${PARTITION}"
echo "  job_name=${JOB_NAME}"
echo "  days=${DAYS}"
echo "  poll_seconds=${POLL_SECONDS}"
echo "  job_name_regex=${JOB_NAME_REGEX}"
echo "  until_epoch=${UNTIL_EPOCH}"

sbatch \
    --partition "${PARTITION}" \
    --job-name "${JOB_NAME}" \
    --export "${EXPORT_STR}" \
    "${SBATCH_FILE}"
