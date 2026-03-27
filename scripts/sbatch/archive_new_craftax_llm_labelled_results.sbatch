#!/bin/bash
#SBATCH --job-name=archive_craftax_llm_results
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/geney/Craftax_Baselines/logs/archive_craftax_llm_results_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/archive_craftax_llm_results_%j.err

set -Eeuo pipefail

SRC_PARENT="${SRC_PARENT:-/data/group_data/rl/geney}"
SRC_NAME="${SRC_NAME:-new_craftax_llm_labelled_results}"
SRC_DIR="${SRC_PARENT}/${SRC_NAME}"
SCRATCH_TAR="${SCRATCH_TAR:-/scratch/geney/${SRC_NAME}.tar.gz}"
DST_TAR="${DST_TAR:-${SRC_PARENT}/${SRC_NAME}.tar.gz}"

log() {
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log "Archive job started on host=$(hostname)"
log "SRC_DIR=${SRC_DIR}"
log "SCRATCH_TAR=${SCRATCH_TAR}"
log "DST_TAR=${DST_TAR}"

mkdir -p "$(dirname "${SCRATCH_TAR}")"

if [[ -f "${DST_TAR}" ]]; then
    log "Destination tar already exists; validating integrity."
    gzip -t "${DST_TAR}"
    log "Destination tar integrity OK."
    if [[ -d "${SRC_DIR}" ]]; then
        log "Deleting source directory because archive is already present."
        rm -rf "${SRC_DIR}"
        log "Deleted source directory."
    else
        log "Source directory already absent."
    fi
    log "Archive job complete (already finalized)."
    exit 0
fi

if [[ -f "${SCRATCH_TAR}" ]]; then
    log "Found existing scratch tar; validating integrity."
    if gzip -t "${SCRATCH_TAR}"; then
        log "Existing scratch tar is valid; reusing it."
    else
        log "Existing scratch tar is invalid (likely partial); deleting."
        rm -f "${SCRATCH_TAR}"
    fi
fi

if [[ ! -f "${SCRATCH_TAR}" ]]; then
    if [[ ! -d "${SRC_DIR}" ]]; then
        log "ERROR: source directory does not exist: ${SRC_DIR}"
        exit 1
    fi
    log "Creating archive at ${SCRATCH_TAR}"
    (
        cd "${SRC_PARENT}"
        tar -czf "${SCRATCH_TAR}" "${SRC_NAME}"
    )
    log "Archive creation finished."
fi

log "Validating scratch archive integrity."
gzip -t "${SCRATCH_TAR}"
log "Scratch archive integrity OK."

log "Moving archive to destination."
if mv "${SCRATCH_TAR}" "${DST_TAR}"; then
    log "Move succeeded."
else
    log "Initial move failed (likely no free space). Deleting source directory then retrying."
    if [[ -d "${SRC_DIR}" ]]; then
        rm -rf "${SRC_DIR}"
        log "Deleted source directory."
    else
        log "Source directory already absent."
    fi
    mv "${SCRATCH_TAR}" "${DST_TAR}"
    log "Move succeeded after source cleanup."
fi

log "Validating destination archive integrity."
gzip -t "${DST_TAR}"
log "Destination archive integrity OK."

if [[ -d "${SRC_DIR}" ]]; then
    log "Deleting source directory."
    rm -rf "${SRC_DIR}"
    log "Deleted source directory."
else
    log "Source directory already absent."
fi

log "Archive workflow completed successfully."
