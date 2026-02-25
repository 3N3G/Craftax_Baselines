#!/bin/bash
set -euo pipefail

# Backward-compatible wrapper.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/babel.sh" logs "$@"
