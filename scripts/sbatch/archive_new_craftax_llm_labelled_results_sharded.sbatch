#!/bin/bash
#SBATCH --job-name=archive_craftax_llm_sharded
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/geney/Craftax_Baselines/logs/archive_craftax_llm_sharded_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/archive_craftax_llm_sharded_%j.err

set -Eeuo pipefail

SRC_PARENT="${SRC_PARENT:-/data/group_data/rl/geney}"
SRC_NAME="${SRC_NAME:-new_craftax_llm_labelled_results}"
SRC_DIR="${SRC_PARENT}/${SRC_NAME}"

# Store compressed shards back on group_data while using user_data as a durable
# temp staging area to avoid giant node-local scratch requirements.
DST_DIR="${DST_DIR:-${SRC_PARENT}/${SRC_NAME}_shards}"
TMP_DIR="${TMP_DIR:-/data/user_data/${USER}/archive_tmp/${SRC_NAME}_shards}"
MANIFEST_TMP="${MANIFEST_TMP:-${TMP_DIR}/manifest.tsv}"
MANIFEST_DST="${MANIFEST_DST:-${DST_DIR}/manifest.tsv}"
LOCK_DIR="/tmp/archive_${SRC_NAME}_sharded.lock"

# Target per-shard uncompressed input budget (default: 40 GiB).
# Keep shards small enough to commit frequently and free source space quickly.
SHARD_INPUT_MAX_GB="${SHARD_INPUT_MAX_GB:-20}"
SHARD_INPUT_MAX_BYTES=$((SHARD_INPUT_MAX_GB * 1024 * 1024 * 1024))
VERIFY_SHARDS="${VERIFY_SHARDS:-0}"

log() {
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

acquire_lock() {
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
        printf "%s\n" "$$" > "${LOCK_DIR}/pid"
        trap 'rm -rf "${LOCK_DIR}"' EXIT
        return 0
    fi

    local holder
    holder="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
    if [[ -n "${holder}" ]] && kill -0 "${holder}" >/dev/null 2>&1; then
        log "Another sharded archive instance is active (pid=${holder}); exiting."
        return 1
    fi

    log "Stale lock detected (pid=${holder:-unknown}); removing lock and retrying."
    rm -rf "${LOCK_DIR}"
    mkdir "${LOCK_DIR}"
    printf "%s\n" "$$" > "${LOCK_DIR}/pid"
    trap 'rm -rf "${LOCK_DIR}"' EXIT
}

ensure_dir() {
    local p="$1"
    if ! mkdir -p "${p}" >/dev/null 2>&1; then
        log "ERROR: unable to create directory: ${p}"
        exit 1
    fi
}

ensure_writable_probe() {
    local p="$1"
    local probe="${p}/.write_probe_${SLURM_JOB_ID:-manual}_$$"
    if ! ( : > "${probe}" ) 2>/dev/null; then
        log "ERROR: directory is not writable: ${p}"
        exit 1
    fi
    rm -f "${probe}" || true
}

verify_shard() {
    local shard="$1"
    if [[ "${VERIFY_SHARDS}" == "1" ]]; then
        gzip -t "${shard}"
    else
        [[ -s "${shard}" ]]
    fi
}

recover_tmp_shards() {
    shopt -s nullglob
    local tmp_shard
    for tmp_shard in "${TMP_DIR}"/shard_*.tar.gz; do
        local base dst_shard
        base="$(basename "${tmp_shard}")"
        dst_shard="${DST_DIR}/${base}"
        if [[ -f "${dst_shard}" ]]; then
            log "Recovered tmp shard already present at destination: ${base}; deleting tmp copy."
            rm -f "${tmp_shard}" "${tmp_shard%.tar.gz}.files" || true
            continue
        fi
        log "Recovering tmp shard to destination: ${base}"
        mv "${tmp_shard}" "${dst_shard}"
        verify_shard "${dst_shard}"
        mv -f "${tmp_shard%.tar.gz}.files" "${DST_DIR}/" 2>/dev/null || true
    done
    shopt -u nullglob
}

next_shard_index() {
    local n=0
    shopt -s nullglob
    for f in "${DST_DIR}"/shard_*.tar.gz "${TMP_DIR}"/shard_*.tar.gz; do
        local b idx
        b="$(basename "${f}")"
        idx="${b#shard_}"
        idx="${idx%.tar.gz}"
        if [[ "${idx}" =~ ^[0-9]+$ ]]; then
            if ((10#${idx} > n)); then
                n=$((10#${idx}))
            fi
        fi
    done
    shopt -u nullglob
    printf "%06d" $((n + 1))
}

archive_batch() {
    local shard_idx="$1"
    shift
    local rel_files=("$@")
    local shard_name="shard_${shard_idx}.tar.gz"
    local tmp_shard="${TMP_DIR}/${shard_name}"
    local dst_shard="${DST_DIR}/${shard_name}"
    local filelist="${TMP_DIR}/shard_${shard_idx}.files"

    if [[ -f "${dst_shard}" ]]; then
        log "Shard already exists at destination: ${shard_name}; skipping creation."
        for rel in "${rel_files[@]}"; do
            if [[ -f "${SRC_PARENT}/${rel}" ]]; then
                rm -f "${SRC_PARENT}/${rel}"
            fi
        done
        return 0
    fi

    : > "${filelist}"
    local rel
    for rel in "${rel_files[@]}"; do
        printf "%s\n" "${rel}" >> "${filelist}"
    done

    log "Creating shard ${shard_name} with ${#rel_files[@]} files."
    if command -v pigz >/dev/null 2>&1; then
        local pigz_threads="${SLURM_CPUS_PER_TASK:-4}"
        tar -I "pigz -1 -p ${pigz_threads}" -cf "${tmp_shard}" -C "${SRC_PARENT}" -T "${filelist}"
    else
        tar -czf "${tmp_shard}" -C "${SRC_PARENT}" -T "${filelist}"
    fi
    verify_shard "${tmp_shard}"

    # Free group_data space before moving shard back.
    for rel in "${rel_files[@]}"; do
        if [[ -f "${SRC_PARENT}/${rel}" ]]; then
            rm -f "${SRC_PARENT}/${rel}"
        fi
    done

    mv "${tmp_shard}" "${dst_shard}"
    verify_shard "${dst_shard}"
    mv -f "${filelist}" "${DST_DIR}/"

    local shard_size
    shard_size="$(stat -c %s "${dst_shard}")"
    for rel in "${rel_files[@]}"; do
        printf "%s\t%s\t%s\n" "${shard_name}" "${shard_size}" "${rel}" >> "${MANIFEST_TMP}"
    done
}

main() {
    acquire_lock || exit 0

    log "Sharded archive job started on host=$(hostname)"
    log "SRC_DIR=${SRC_DIR}"
    log "DST_DIR=${DST_DIR}"
    log "TMP_DIR=${TMP_DIR}"
    log "SHARD_INPUT_MAX_GB=${SHARD_INPUT_MAX_GB}"
    log "VERIFY_SHARDS=${VERIFY_SHARDS}"

    if [[ ! -d "${SRC_DIR}" ]]; then
        log "Source directory missing; nothing to archive."
        exit 0
    fi

    ensure_dir "${DST_DIR}"
    ensure_dir "${TMP_DIR}"
    ensure_writable_probe "${TMP_DIR}"
    touch "${MANIFEST_TMP}"

    recover_tmp_shards

    local -a batch_files=()
    local batch_bytes=0
    local rel size
    local pending_raw="${TMP_DIR}/pending_files_${SLURM_JOB_ID:-manual}_$$.tsv"

    find "${SRC_DIR}" -maxdepth 1 -type f -name 'trajectories_batch_*.npz' -printf '%f\t%s\n' \
        | sort > "${pending_raw}"

    while IFS=$'\t' read -r rel size; do
        rel="${SRC_NAME}/${rel}"
        # Start a new shard if the next file would exceed the target batch size.
        if (( batch_bytes > 0 && batch_bytes + size > SHARD_INPUT_MAX_BYTES )); then
            archive_batch "$(next_shard_index)" "${batch_files[@]}"
            batch_files=()
            batch_bytes=0
        fi

        batch_files+=("${rel}")
        batch_bytes=$((batch_bytes + size))
    done < "${pending_raw}"
    rm -f "${pending_raw}"

    if (( ${#batch_files[@]} > 0 )); then
        archive_batch "$(next_shard_index)" "${batch_files[@]}"
    fi

    if find "${SRC_DIR}" -mindepth 1 -print -quit | grep -q .; then
        log "WARNING: source directory still has files after sharding."
    else
        rmdir "${SRC_DIR}" 2>/dev/null || true
        log "Source directory emptied and removed."
    fi

    cp -f "${MANIFEST_TMP}" "${MANIFEST_DST}" 2>/dev/null || true

    log "Sharded archive workflow completed."
}

main "$@"
