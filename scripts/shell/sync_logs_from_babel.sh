#!/bin/bash
set -euo pipefail

# Sync logs from babel -> local logs/.
#
# Usage:
#   scripts/shell/sync_logs_from_babel.sh
#   scripts/shell/sync_logs_from_babel.sh <jobid>
#
# No args: pull all *.out/*.err/*.log/*.json from remote logs/ plus slurm-*.out/err.
# jobid: pull only files containing that job id.

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
REMOTE_HOST="${BABEL_HOST:-babel}"
REMOTE_ROOT="${BABEL_ROOT:-~/Craftax_Baselines}"
JOBID="${1:-}"

mkdir -p "${ROOT}/logs" "${ROOT}/logs/slurm_home"

if [[ -z "$JOBID" ]]; then
    echo "Syncing all remote logs..."
    rsync -avz --update \
        --include "*.out" \
        --include "*.err" \
        --include "*.log" \
        --include "*.json" \
        --exclude "*" \
        "${REMOTE_HOST}:${REMOTE_ROOT}/logs/" "${ROOT}/logs/"

    rsync -avz --update \
        --include "slurm-*.out" \
        --include "slurm-*.err" \
        --exclude "*" \
        "${REMOTE_HOST}:~/" "${ROOT}/logs/slurm_home/"
else
    echo "Syncing remote logs for jobid=${JOBID}..."
    rsync -avz --update \
        --include "*${JOBID}*.out" \
        --include "*${JOBID}*.err" \
        --include "*${JOBID}*.log" \
        --include "*${JOBID}*.json" \
        --exclude "*" \
        "${REMOTE_HOST}:${REMOTE_ROOT}/logs/" "${ROOT}/logs/"

    rsync -avz --update \
        --include "slurm-${JOBID}.out" \
        --include "slurm-${JOBID}.err" \
        --exclude "*" \
        "${REMOTE_HOST}:~/" "${ROOT}/logs/slurm_home/"
fi

echo "Done."
