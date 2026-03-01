#!/bin/bash
#SBATCH --job-name=cot_promote_selector
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cot_promote_selector_%j.out
#SBATCH --error=logs/cot_promote_selector_%j.err

set -euo pipefail

source ~/.bashrc || true
cd ~/Craftax_Baselines
mkdir -p logs

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found"
    exit 1
fi
conda activate /data/user_data/geney/.conda/envs/craftax || conda activate craftax || true

MANIFEST="${1:?manifest path required}"
TOP_K="${2:-2}"
PROMOTE_TARGET="${3:-300000000}"
STAGE_TARGET="${4:-60000000}"

python -u scripts/select_and_submit_cot_promotions.py \
    --manifest "${MANIFEST}" \
    --top-k "${TOP_K}" \
    --promote-target "${PROMOTE_TARGET}" \
    --stage-target "${STAGE_TARGET}"
