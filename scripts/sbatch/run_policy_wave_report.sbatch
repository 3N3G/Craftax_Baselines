#!/bin/bash
#SBATCH --job-name=wavev2_report
#SBATCH --partition=rl
#SBATCH --qos=rl_qos
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

source scripts/sbatch/conda_env_setup.sh
sbatch_require_conda
REPORT_ENV_PATH="${REPORT_ENV_PATH:-/data/user_data/geney/.conda/envs/craftax}"
sbatch_activate_conda_env "policy-wave-report" "${REPORT_ENV_PATH}"

SUMMARY_JSON="${SUMMARY_JSON:-analysis/policy_wave_v2/policy_wave_v2_summary.json}"
OUTPUT_MD="${OUTPUT_MD:-analysis/reports/policy_wave_v2_report.md}"

python -u analysis/reports/generate_policy_wave_v2_report.py \
    --summary_json "${SUMMARY_JSON}" \
    --output_md "${OUTPUT_MD}"
