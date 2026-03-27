#!/bin/bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Optional per-user config (not committed by default)
if [[ -f "${ROOT}/.babel.env" ]]; then
    # shellcheck disable=SC1090
    source "${ROOT}/.babel.env"
fi

BABEL_HOST="${BABEL_HOST:-babel}"
BABEL_USER="${BABEL_USER:-}"
BABEL_ROOT="${BABEL_ROOT:-~/Craftax_Baselines}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${ROOT}/logs}"
LOCAL_RUNTIME_LOG_DIR="${LOCAL_RUNTIME_LOG_DIR:-${LOCAL_LOG_DIR}/runtime}"

if [[ -n "${BABEL_USER}" ]]; then
    BABEL_LOGIN="${BABEL_USER}@${BABEL_HOST}"
else
    BABEL_LOGIN="${BABEL_HOST}"
fi

rsync_base_args=( -avz )
rsync_excludes=(
    --exclude ".git"
    --exclude "__pycache__/"
    --exclude "*.pyc"
    --exclude ".DS_Store"
    --exclude "wandb/"
    --exclude "logs/"
    --exclude "slurm_home/"
    --exclude "tmp/"
)

usage() {
    cat <<'EOF'
Unified Babel operations helper.

Usage:
  scripts/shell/babel.sh config
  scripts/shell/babel.sh push [--all] [--delete] [--dry-run] [path ...]
  scripts/shell/babel.sh pull [--all] [--dry-run] [path ...]
  scripts/shell/babel.sh logs [jobid]
  scripts/shell/babel.sh squeue [raw squeue args...]
  scripts/shell/babel.sh ssh [command...]
  scripts/shell/babel.sh sbatch <args...>

Defaults:
  BABEL_HOST   (default: babel)
  BABEL_USER   (optional)
  BABEL_ROOT   (default: ~/Craftax_Baselines)
  LOCAL_LOG_DIR(default: <repo>/logs)
  LOCAL_RUNTIME_LOG_DIR(default: <repo>/logs/runtime)

Notes:
  - `push` with no paths behaves as `push --all`.
  - `pull` is conservative; use paths or `--all`.
  - `logs/`, `slurm_home/`, `tmp/`, and `wandb/` are excluded from full push/pull.
  - `--delete` is blocked unless BABEL_ALLOW_DELETE=1 is set.
  - For reproducible runs, submit via sbatch wrappers under scripts/sbatch.
EOF
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $1"
        exit 1
    fi
}

print_config() {
    cat <<EOF
ROOT=${ROOT}
BABEL_LOGIN=${BABEL_LOGIN}
BABEL_ROOT=${BABEL_ROOT}
LOCAL_LOG_DIR=${LOCAL_LOG_DIR}
LOCAL_RUNTIME_LOG_DIR=${LOCAL_RUNTIME_LOG_DIR}
EOF
}

rel_path_from_input() {
    local p="$1"
    local abs=""

    if [[ "$p" == /* ]]; then
        abs="$p"
    elif [[ -e "${ROOT}/$p" ]]; then
        abs="${ROOT}/$p"
    elif [[ -e "$p" ]]; then
        abs="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
    else
        # For pull, local path might not exist yet. Return normalized relative path.
        p="${p#./}"
        if [[ "$p" == "" ]]; then
            echo "."
        else
            echo "$p"
        fi
        return 0
    fi

    case "$abs" in
        "${ROOT}")
            echo "."
            ;;
        "${ROOT}"/*)
            echo "${abs#${ROOT}/}"
            ;;
        *)
            echo "ERROR: path is outside repo root: $p" >&2
            return 1
            ;;
    esac
}

quote_join() {
    local out=""
    local a
    for a in "$@"; do
        out+=" $(printf '%q' "$a")"
    done
    echo "$out"
}

cmd_push() {
    require_cmd rsync

    local do_all=0
    local do_delete=0
    local dry_run=0
    local args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --all)
                do_all=1
                shift
                ;;
            --delete)
                do_delete=1
                shift
                ;;
            --dry-run)
                dry_run=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                args+=("$1")
                shift
                ;;
        esac
    done

    if [[ ${#args[@]} -eq 0 && $do_all -eq 0 ]]; then
        do_all=1
    fi

    if [[ $do_delete -eq 1 && "${BABEL_ALLOW_DELETE:-0}" != "1" ]]; then
        echo "ERROR: refusing --delete without BABEL_ALLOW_DELETE=1"
        echo "Set BABEL_ALLOW_DELETE=1 only for intentional cleanup syncs."
        exit 2
    fi

    if [[ $do_all -eq 1 ]]; then
        local rsync_args=("${rsync_base_args[@]}" "${rsync_excludes[@]}")
        if [[ $do_delete -eq 1 ]]; then
            rsync_args+=(--delete)
        fi
        if [[ $dry_run -eq 1 ]]; then
            rsync_args+=(--dry-run)
        fi
        echo "[push] syncing full repo -> ${BABEL_LOGIN}:${BABEL_ROOT}"
        rsync "${rsync_args[@]}" "${ROOT}/" "${BABEL_LOGIN}:${BABEL_ROOT}/"
        return
    fi

    local p rel rsync_args
    for p in "${args[@]}"; do
        rel="$(rel_path_from_input "$p")"
        if [[ ! -e "${ROOT}/${rel}" ]]; then
            echo "ERROR: local path not found for push: ${p}"
            exit 1
        fi
        rsync_args=("${rsync_base_args[@]}" --relative)
        if [[ $dry_run -eq 1 ]]; then
            rsync_args+=(--dry-run)
        fi
        echo "[push] ${rel}"
        (
            cd "${ROOT}"
            rsync "${rsync_args[@]}" "${rel}" "${BABEL_LOGIN}:${BABEL_ROOT}/."
        )
    done
}

cmd_pull() {
    require_cmd rsync

    local do_all=0
    local dry_run=0
    local args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --all)
                do_all=1
                shift
                ;;
            --dry-run)
                dry_run=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                args+=("$1")
                shift
                ;;
        esac
    done

    if [[ ${#args[@]} -eq 0 && $do_all -eq 0 ]]; then
        echo "ERROR: pull requires paths or --all"
        exit 2
    fi

    if [[ $do_all -eq 1 ]]; then
        local rsync_args=("${rsync_base_args[@]}" "${rsync_excludes[@]}")
        if [[ $dry_run -eq 1 ]]; then
            rsync_args+=(--dry-run)
        fi
        echo "[pull] syncing full repo <- ${BABEL_LOGIN}:${BABEL_ROOT}"
        rsync "${rsync_args[@]}" "${BABEL_LOGIN}:${BABEL_ROOT}/" "${ROOT}/"
        return
    fi

    local p rel rsync_args
    for p in "${args[@]}"; do
        rel="$(rel_path_from_input "$p")"
        rsync_args=("${rsync_base_args[@]}" --relative)
        if [[ $dry_run -eq 1 ]]; then
            rsync_args+=(--dry-run)
        fi
        echo "[pull] ${rel}"
        rsync "${rsync_args[@]}" "${BABEL_LOGIN}:${BABEL_ROOT}/./${rel}" "${ROOT}/"
    done
}

cmd_logs() {
    require_cmd rsync
    local jobid="${1:-}"
    local remote_user=""
    local remote_runtime_dir=""
    local have_runtime_logs=0

    mkdir -p "${LOCAL_LOG_DIR}" "${LOCAL_LOG_DIR}/slurm_home" "${LOCAL_RUNTIME_LOG_DIR}"
    remote_user="$(ssh "${BABEL_LOGIN}" "whoami" 2>/dev/null | tr -d '\r')"
    if [[ -n "${remote_user}" ]]; then
        remote_runtime_dir="/data/user_data/${remote_user}/craftax_job_logs"
        if ssh "${BABEL_LOGIN}" "test -d '${remote_runtime_dir}'" >/dev/null 2>&1; then
            have_runtime_logs=1
        fi
    fi

    if [[ -z "$jobid" ]]; then
        echo "[logs] syncing all logs"
        rsync -avz --update \
            --include "*.out" \
            --include "*.err" \
            --include "*.log" \
            --include "*.json" \
            --exclude "*" \
            "${BABEL_LOGIN}:${BABEL_ROOT}/logs/" "${LOCAL_LOG_DIR}/"

        rsync -avz --update \
            --include "slurm-*.out" \
            --include "slurm-*.err" \
            --exclude "*" \
            "${BABEL_LOGIN}:~/" "${LOCAL_LOG_DIR}/slurm_home/"

        if [[ ${have_runtime_logs} -eq 1 ]]; then
            rsync -avz --update \
                --include "*.out" \
                --include "*.err" \
                --include "*.log" \
                --exclude "*" \
                "${BABEL_LOGIN}:${remote_runtime_dir}/" "${LOCAL_RUNTIME_LOG_DIR}/"
        fi
    else
        echo "[logs] syncing logs for jobid=${jobid}"
        rsync -avz --update \
            --include "*${jobid}*.out" \
            --include "*${jobid}*.err" \
            --include "*${jobid}*.log" \
            --include "*${jobid}*.json" \
            --exclude "*" \
            "${BABEL_LOGIN}:${BABEL_ROOT}/logs/" "${LOCAL_LOG_DIR}/"

        rsync -avz --update \
            --include "slurm-${jobid}.out" \
            --include "slurm-${jobid}.err" \
            --exclude "*" \
            "${BABEL_LOGIN}:~/" "${LOCAL_LOG_DIR}/slurm_home/"

        if [[ ${have_runtime_logs} -eq 1 ]]; then
            rsync -avz --update \
                --include "*${jobid}*.out" \
                --include "*${jobid}*.err" \
                --include "*${jobid}*.log" \
                --exclude "*" \
                "${BABEL_LOGIN}:${remote_runtime_dir}/" "${LOCAL_RUNTIME_LOG_DIR}/"
        fi
    fi
}

cmd_squeue() {
    if [[ $# -eq 0 ]]; then
        ssh "${BABEL_LOGIN}" "squeue -u \$USER -o '%.18i %.9P %.24j %.8T %.10M %.10l %.6D %R'"
    else
        local q
        q="$(quote_join "$@")"
        ssh "${BABEL_LOGIN}" "squeue${q}"
    fi
}

cmd_ssh() {
    if [[ $# -eq 0 ]]; then
        ssh -t "${BABEL_LOGIN}" "cd ${BABEL_ROOT} && exec bash -l"
    else
        local q
        q="$(quote_join "$@")"
        ssh "${BABEL_LOGIN}" "cd ${BABEL_ROOT} &&${q}"
    fi
}

cmd_sbatch() {
    if [[ $# -lt 1 ]]; then
        echo "ERROR: sbatch requires arguments"
        exit 2
    fi
    local q
    q="$(quote_join "$@")"
    ssh "${BABEL_LOGIN}" "cd ${BABEL_ROOT} && sbatch${q}"
}

main() {
    local subcmd="${1:-}"
    if [[ -z "$subcmd" ]]; then
        usage
        exit 2
    fi
    shift || true

    case "$subcmd" in
        config)
            print_config
            ;;
        push)
            cmd_push "$@"
            ;;
        pull)
            cmd_pull "$@"
            ;;
        logs)
            cmd_logs "$@"
            ;;
        squeue|jobs)
            cmd_squeue "$@"
            ;;
        ssh)
            cmd_ssh "$@"
            ;;
        sbatch)
            cmd_sbatch "$@"
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            echo "ERROR: unknown subcommand: $subcmd"
            usage
            exit 2
            ;;
    esac
}

main "$@"
