#!/bin/bash
#SBATCH --job-name=rtg_exact
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/geney/Craftax_Baselines/logs/rtg_exact_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/rtg_exact_%j.err

set -eo pipefail

echo "Starting exact RTG recompute on $(hostname) at $(date)"
cd /home/geney/Craftax_Baselines

source ~/.bashrc >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
    conda activate /data/user_data/geney/.conda/envs/craftax_fast_llm >/dev/null 2>&1 \
        || conda activate craftax_fast_llm >/dev/null 2>&1 \
        || true
fi

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python not found after environment setup."
    exit 1
fi

DATA_DIR="${DATA_DIR:-/data/group_data/rl/geney/vllm_craftax_labelled_results}"
SEARCH_PATTERN="${SEARCH_PATTERN:-trajectories_batch_*.npz}"
NUM_ENVS="${NUM_ENVS:-128}"
GAMMA="${GAMMA:-0.99}"
START_BATCH="${START_BATCH:-}"
END_BATCH="${END_BATCH:-}"

CMD=(
    python tools/ppo_add_returns_exact.py
    --data_dir "${DATA_DIR}"
    --search_pattern "${SEARCH_PATTERN}"
    --num_envs "${NUM_ENVS}"
    --gamma "${GAMMA}"
)

if [[ -n "${START_BATCH}" ]]; then
    CMD+=(--start_batch "${START_BATCH}")
fi
if [[ -n "${END_BATCH}" ]]; then
    CMD+=(--end_batch "${END_BATCH}")
fi

echo "Running:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Exact RTG recompute finished at $(date)"
