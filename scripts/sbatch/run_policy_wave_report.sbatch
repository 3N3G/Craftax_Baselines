#!/bin/bash
#SBATCH --job-name=wavev2_report
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/geney/Craftax_Baselines/logs/wavev2_report_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/wavev2_report_%j.err

set -euo pipefail

set +e
set +u
set +o pipefail
source ~/.bashrc
BASHRC_RC=$?
set -e
set -u
set -o pipefail
if [[ $BASHRC_RC -ne 0 ]]; then
    echo "WARNING: source ~/.bashrc exited with code $BASHRC_RC; continuing"
fi

cd ~/Craftax_Baselines
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if command -v conda >/dev/null 2>&1; then
    conda activate /data/user_data/geney/.conda/envs/craftax >/dev/null 2>&1 \
        || conda activate craftax >/dev/null 2>&1 \
        || true
fi

SUMMARY_JSON="${SUMMARY_JSON:-analysis/policy_wave_v2/policy_wave_v2_summary.json}"
OUTPUT_MD="${OUTPUT_MD:-analysis/reports/policy_wave_v2_report.md}"

python -u analysis/reports/generate_policy_wave_v2_report.py \
    --summary_json "${SUMMARY_JSON}" \
    --output_md "${OUTPUT_MD}"

