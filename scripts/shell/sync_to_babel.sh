#!/bin/bash
set -euo pipefail

# Backward-compatible wrapper.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 <file_or_dir> [more_paths...]"
    exit 2
fi

exec "${SCRIPT_DIR}/babel.sh" push "$@"
