#!/bin/bash

# Shared conda activation helpers for sbatch entrypoints.
# These helpers intentionally avoid candidate fallback lists so each workflow
# declares one explicit env path.

sbatch_require_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found after sourcing bashrc"
        return 1
    fi
}

sbatch_activate_conda_env() {
    local purpose="$1"
    local env_path="$2"

    if [[ -z "${env_path}" ]]; then
        echo "ERROR: empty env path for ${purpose}"
        return 1
    fi

    if [[ ! -d "${env_path}" ]]; then
        echo "ERROR: conda env path does not exist for ${purpose}: ${env_path}"
        conda env list || true
        return 1
    fi

    if ! conda activate "${env_path}" >/dev/null 2>&1; then
        echo "ERROR: failed to activate conda env for ${purpose}: ${env_path}"
        conda env list || true
        return 1
    fi

    echo "Activated conda env for ${purpose}: ${env_path}"
    return 0
}
