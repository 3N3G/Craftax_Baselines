#!/bin/bash
#SBATCH --job-name=eval_offline_llm
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/geney/Craftax_Baselines/logs/eval_offline_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/eval_offline_%j.err

set -eo pipefail

cd /home/geney/Craftax_Baselines
source ~/.bashrc >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
    conda activate craftax_fast_llm >/dev/null 2>&1 \
        || conda activate /data/user_data/geney/.conda/envs/craftax_fast_llm >/dev/null 2>&1 \
        || true
fi

CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
    echo "ERROR: CHECKPOINT env var is required."
    exit 1
fi

NUM_EPISODES="${NUM_EPISODES:-64}"
NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-42}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-20000}"
HIDDEN_INPUT_MODE="${HIDDEN_INPUT_MODE:-zero}"  # zero | llm
SKIP_N="${SKIP_N:-1}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
NO_WANDB="${NO_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-craftax_symbolic_evals}"
WANDB_ENTITY="${WANDB_ENTITY:-iris-sobolmark}"
WANDB_NAME="${WANDB_NAME:-}"

VLLM_PID=""
cleanup() {
    if [[ -n "${VLLM_PID}" ]]; then
        kill "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "${HIDDEN_INPUT_MODE}" == "llm" ]]; then
    echo "Starting vLLM server for LLM hidden-state evaluation..."
    bash scripts/start_vllm_hidden.sh --mode last_token > "logs/vllm_eval_offline_${SLURM_JOB_ID}.log" 2>&1 &
    VLLM_PID=$!
    for i in $(seq 1 240); do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            echo "vLLM ready."
            break
        fi
        sleep 1
    done
    curl -sf http://localhost:8000/health >/dev/null
fi

CMD=(
    python scripts/eval_offline_llm_symbolic.py
    --checkpoint "${CHECKPOINT}"
    --hidden-input-mode "${HIDDEN_INPUT_MODE}"
    --skip-n "${SKIP_N}"
    --num-episodes "${NUM_EPISODES}"
    --num-envs "${NUM_ENVS}"
    --seed "${SEED}"
    --max-env-steps "${MAX_ENV_STEPS}"
    --device "${DEVICE}"
)

if [[ -n "${OUTPUT_JSON}" ]]; then
    CMD+=(--output-json "${OUTPUT_JSON}")
fi
if [[ "${NO_WANDB}" == "1" ]]; then
    CMD+=(--no-wandb)
else
    CMD+=(--wandb-project "${WANDB_PROJECT}" --wandb-entity "${WANDB_ENTITY}")
    if [[ -n "${WANDB_NAME}" ]]; then
        CMD+=(--wandb-name "${WANDB_NAME}")
    fi
fi

echo "Running eval command:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"
