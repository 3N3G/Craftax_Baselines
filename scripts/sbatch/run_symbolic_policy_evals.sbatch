#!/bin/bash
#SBATCH --job-name=symbolic_evals
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/symbolic_evals_%j.out
#SBATCH --error=logs/symbolic_evals_%j.err

set -euo pipefail

# Source bashrc defensively.
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
mkdir -p logs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found after sourcing bashrc"
    exit 1
fi

activate_conda_for() {
    local purpose="$1"
    shift
    local env_name
    for env_name in "$@"; do
        if conda activate "${env_name}" >/dev/null 2>&1; then
            echo "Activated conda env for ${purpose}: ${env_name}"
            return 0
        fi
    done
    echo "ERROR: failed to activate conda env for ${purpose}"
    echo "Tried: $*"
    conda env list || true
    return 1
}

echo "========================================"
echo "Craftax Symbolic Policy Eval Suite"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "========================================"

NUM_ENVS=${NUM_ENVS:-128}
TARGET_EPISODES=${TARGET_EPISODES:-128}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-80000}
OUTPUT_JSON=${OUTPUT_JSON:-logs/policy_eval_craftax_symbolic_evals_${SLURM_JOB_ID}.json}
LLM_TOKENS=${LLM_TOKENS:-1}
VIDEO_MAX_STEPS=${VIDEO_MAX_STEPS:-600}
VIDEO_FPS=${VIDEO_FPS:-12}
VIDEO_LLM_TOKENS=${VIDEO_LLM_TOKENS:-1}
VIDEO_LLM_MAX_WORDS=${VIDEO_LLM_MAX_WORDS:-12}
VIDEO_GAMMA=${VIDEO_GAMMA:-0.99}
VLLM_SAFE_MODE=${VLLM_SAFE_MODE:-0}
START_VLLM=${START_VLLM:-1}

# Keep vLLM cache/tmp off shared home and onto local scratch.
export VLLM_TMP_ROOT=${VLLM_TMP_ROOT:-/scratch/${USER}/craftax_vllm_eval_${SLURM_JOB_ID}}
mkdir -p "${VLLM_TMP_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${VLLM_TMP_ROOT}/pycache"

echo "Scratch storage:"
df -h /scratch || true

VLLM_PID=""
cleanup() {
    if [[ -n "${VLLM_PID}" ]]; then
        echo "Stopping vLLM server (pid=${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "${START_VLLM}" == "1" ]]; then
    echo "Starting vLLM hidden-state server..."
    activate_conda_for "vllm-server" \
        /data/user_data/geney/.conda/envs/craftax_fast_llm \
        craftax_fast_llm \
        base

    VLLM_ARGS=(--mode last_token)
    if [[ "${VLLM_SAFE_MODE}" == "1" ]]; then
        VLLM_ARGS+=(--safe)
    fi
    bash scripts/start_vllm_hidden.sh "${VLLM_ARGS[@]}" > "logs/vllm_symbolic_eval_${SLURM_JOB_ID}.log" 2>&1 &
    VLLM_PID=$!

    echo "Waiting for vLLM /health..."
    SECONDS=0
    while true; do
        if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
            echo "vLLM ready after ${SECONDS}s"
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM server process died."
            tail -100 "logs/vllm_symbolic_eval_${SLURM_JOB_ID}.log" || true
            exit 1
        fi
        if [[ ${SECONDS} -gt 1200 ]]; then
            echo "ERROR: timed out waiting for vLLM to become healthy."
            tail -100 "logs/vllm_symbolic_eval_${SLURM_JOB_ID}.log" || true
            exit 1
        fi
        sleep 2
    done
else
    echo "START_VLLM=0: skipping vLLM startup."
fi

echo "Switching to training/eval env..."
activate_conda_for "policy-eval" \
    /data/user_data/geney/.conda/envs/craftax \
    craftax \
    base

echo "Running evaluation suite..."
python -u scripts/eval_symbolic_policy_suite.py \
    --wandb-project craftax_symbolic_evals \
    --wandb-entity iris-sobolmark \
    --num-envs "${NUM_ENVS}" \
    --target-episodes "${TARGET_EPISODES}" \
    --max-env-steps "${MAX_ENV_STEPS}" \
    --llm-tokens "${LLM_TOKENS}" \
    --video-max-steps "${VIDEO_MAX_STEPS}" \
    --video-fps "${VIDEO_FPS}" \
    --video-llm-tokens "${VIDEO_LLM_TOKENS}" \
    --video-llm-max-words "${VIDEO_LLM_MAX_WORDS}" \
    --video-gamma "${VIDEO_GAMMA}" \
    --output-json "${OUTPUT_JSON}" \
    "$@"

echo "Done: $(date)"
echo "Output JSON: ${OUTPUT_JSON}"
