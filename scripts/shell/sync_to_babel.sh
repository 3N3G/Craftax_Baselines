#!/bin/bash
set -euo pipefail

# Sync selected files from local repo -> babel:~/Craftax_Baselines
# Usage:
#   scripts/shell/sync_to_babel.sh path/to/file1 path/to/file2 ...

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
REMOTE_HOST="${BABEL_HOST:-babel}"
REMOTE_ROOT="${BABEL_ROOT:-~/Craftax_Baselines}"

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 <file_or_dir> [more_paths...]"
    exit 2
fi

cd "$ROOT"

for p in "$@"; do
    if [[ "$p" = /* ]]; then
        rel="${p#${ROOT}/}"
    else
        rel="$p"
    fi

    if [[ ! -e "$rel" ]]; then
        echo "ERROR: path not found in repo: $p"
        exit 1
    fi

    echo "Syncing: $rel"
    rsync -avz --relative "$rel" "${REMOTE_HOST}:${REMOTE_ROOT}/."
done

echo "Done."
