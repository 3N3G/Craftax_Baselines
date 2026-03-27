#!/bin/bash
#SBATCH --job-name=eval_bundle_llm
#SBATCH --partition=rl
#SBATCH --qos=rl_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/geney/Craftax_Baselines/logs/eval_bundle_llm_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/eval_bundle_llm_%j.err

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
EVAL_ENV_PATH="${EVAL_ENV_PATH:-/data/user_data/geney/.conda/envs/imaug}"
sbatch_activate_conda_env "eval-bundle-llm" "${EVAL_ENV_PATH}"

MANIFEST="${MANIFEST:-configs/eval/policy_wave_v2.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/policy_wave_v2/bundle_reactions}"
BUNDLE_DIRS="${BUNDLE_DIRS:-}"
LLM_GENERATION_TOKENS="${LLM_GENERATION_TOKENS:-32}"
NO_WANDB="${NO_WANDB:-0}"

if [[ -z "${BUNDLE_DIRS}" ]]; then
    echo "ERROR: BUNDLE_DIRS must be provided"
    exit 1
fi

CMD=(
    python -u scripts/eval_llm_bundle_reactions.py
    --manifest "${MANIFEST}"
    --output_dir "${OUTPUT_DIR}"
    --bundle_dirs "${BUNDLE_DIRS}"
    --llm_generation_tokens "${LLM_GENERATION_TOKENS}"
)
if [[ "${NO_WANDB}" == "1" ]]; then
    CMD+=(--no_wandb)
fi

printf '  %q' "${CMD[@]}"
echo
"${CMD[@]}"
