#!/bin/bash
set -Eeuo pipefail

TAR_PATH="/scratch/geney/new_craftax_llm_labelled_results.tar.gz"
DST_DIR="/data/group_data/rl/geney"
DST_TAR="${DST_DIR}/new_craftax_llm_labelled_results.tar.gz"
SRC_DIR="/data/group_data/rl/geney/new_craftax_llm_labelled_results"
POLL_SECONDS="${POLL_SECONDS:-120}"
STABLE_WAIT_SECONDS="${STABLE_WAIT_SECONDS:-60}"
LOCK_DIR="/tmp/finalize_new_craftax_llm_labelled_results.lock"

log() {
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log "Finalizer started on host=$(hostname)"
log "TAR_PATH=${TAR_PATH}"
log "DST_TAR=${DST_TAR}"
log "SRC_DIR=${SRC_DIR}"

acquire_lock() {
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
        printf "%s\n" "$$" > "${LOCK_DIR}/pid"
        trap 'rm -rf "${LOCK_DIR}"' EXIT
        return 0
    fi

    holder="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
    if [[ -n "${holder}" ]] && kill -0 "${holder}" >/dev/null 2>&1; then
        log "Another finalizer instance is active (pid=${holder}); exiting."
        return 1
    fi

    log "Stale lock detected (pid=${holder:-unknown}); removing lock and retrying."
    rm -rf "${LOCK_DIR}"
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
        printf "%s\n" "$$" > "${LOCK_DIR}/pid"
        trap 'rm -rf "${LOCK_DIR}"' EXIT
        return 0
    fi

    holder="$(cat "${LOCK_DIR}/pid" 2>/dev/null || echo unknown)"
    log "Failed to acquire lock after stale-lock cleanup (pid=${holder}); exiting."
    return 1
}

if ! acquire_lock; then
    exit 0
fi

if [[ -f "${DST_TAR}" ]]; then
    log "Destination tarball already exists."
    log "Validating existing destination archive integrity..."
    if ! gzip -t "${DST_TAR}"; then
        log "ERROR: destination archive failed integrity check. Refusing to delete source."
        exit 1
    fi
    if [[ -d "${SRC_DIR}" ]]; then
        log "Deleting leftover source directory: ${SRC_DIR}"
        rm -rf "${SRC_DIR}"
        log "Deleted source directory"
    fi
    log "Finalizer complete (already finalized)."
    exit 0
fi

if [[ ! -e "${TAR_PATH}" ]]; then
    log "Waiting for tarball to appear..."
fi
while [[ ! -e "${TAR_PATH}" ]]; do
    sleep "${POLL_SECONDS}"
    log "Still waiting for tarball..."
done

# Wait for the original tar command to finish.
while pgrep -f "tar -czvf ${TAR_PATH} new_craftax_llm_labelled_results" >/dev/null 2>&1; do
    log "Tar process still running; waiting ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
done

# Ensure file size has stabilized.
while true; do
    s1="$(stat -c %s "${TAR_PATH}")"
    sleep "${STABLE_WAIT_SECONDS}"
    s2="$(stat -c %s "${TAR_PATH}")"
    if [[ "${s1}" == "${s2}" ]]; then
        log "Tarball size stable at ${s2} bytes"
        break
    fi
    log "Tarball still growing (${s1} -> ${s2}); waiting"
done

log "Validating archive integrity (gzip -t)..."
if gzip -t "${TAR_PATH}"; then
    log "Archive integrity check passed"
else
    log "ERROR: archive integrity check failed. Refusing to move/delete."
    exit 1
fi

mkdir -p "${DST_DIR}"

log "Attempting move: ${TAR_PATH} -> ${DST_TAR}"
if mv "${TAR_PATH}" "${DST_TAR}"; then
    log "Move succeeded"
else
    log "Initial move failed (likely no space). Deleting source directory, then retrying move."
    if [[ -d "${SRC_DIR}" ]]; then
        rm -rf "${SRC_DIR}"
        log "Deleted source directory after move failure: ${SRC_DIR}"
    else
        log "Source directory already missing: ${SRC_DIR}"
    fi
    mv "${TAR_PATH}" "${DST_TAR}"
    log "Move succeeded after source directory cleanup"
fi

if [[ -d "${SRC_DIR}" ]]; then
    log "Deleting source directory: ${SRC_DIR}"
    rm -rf "${SRC_DIR}"
    log "Deleted source directory"
else
    log "Source directory already absent; nothing to delete"
fi

if [[ -f "${DST_TAR}" ]]; then
    log "Finalizer complete. Tarball available at ${DST_TAR}"
else
    log "ERROR: expected tarball missing at destination"
    exit 1
fi
