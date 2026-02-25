#!/bin/bash
#SBATCH --job-name=online_rl_hidden_jax
#SBATCH --partition=rl
#SBATCH --gres=gpu:RTX_PRO_6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/online_rl_hidden_jax_%j.out
#SBATCH --error=logs/online_rl_hidden_jax_%j.err

# Online RL with vLLM Hidden States - JAX implementation.

set -Eeo pipefail

# Source bashrc defensively because some profile scripts can return non-zero.
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

mkdir -p logs
cd ~/Craftax_Baselines
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Mirror runtime logs to a non-repo path so accidental repo sync/delete
# cannot erase the only diagnostics for a failed run.
RUNTIME_LOG_ROOT="${RUNTIME_LOG_ROOT:-/data/user_data/${USER}/craftax_job_logs}"
mkdir -p "${RUNTIME_LOG_ROOT}" >/dev/null 2>&1 || true
RUNTIME_STDOUT="${RUNTIME_LOG_ROOT}/online_rl_hidden_jax_${SLURM_JOB_ID}.runtime.out"
RUNTIME_STDERR="${RUNTIME_LOG_ROOT}/online_rl_hidden_jax_${SLURM_JOB_ID}.runtime.err"
exec > >(tee -a "${RUNTIME_STDOUT}")
exec 2> >(tee -a "${RUNTIME_STDERR}" >&2)
echo "Runtime log mirror: ${RUNTIME_STDOUT}"
echo "Runtime error mirror: ${RUNTIME_STDERR}"

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

# vLLM server needs craftax_fast_llm, training needs craftax (for distrax)
conda activate craftax_fast_llm

# Parameters
ENVS=${1:-8}
TIMESTEPS=${2:-1000000}
SKIP_N=${3:-1}
LAYER=${4:--1}  # Which layer to extract (-1 for last available)
TOKENS=${5:-1}  # How many tokens to generate (1 for prompt-only)
NUM_STEPS=${6:-64}
CHECKPOINT_EVERY_STEPS=${7:-10000000}
POLICY_SAVE_DIR=${8:-/data/group_data/rl/geney/online_rl_hidden_models}
CHECKPOINT_DIR=${9:-}
RESUME_FROM=${10:-}
FUSION_MODE=${FUSION_MODE:-concat_raw}
HIDDEN_GATE_INIT_LOGIT=${HIDDEN_GATE_INIT_LOGIT:--4.0}
ACTOR_HEAD_LAYERS=${ACTOR_HEAD_LAYERS:-1}
CRITIC_HEAD_LAYERS=${CRITIC_HEAD_LAYERS:-2}
NO_LLM=${NO_LLM:-0}
RUN_NAME=${RUN_NAME:-}
VLLM_PORT_BASE=${VLLM_PORT_BASE:-18000}
VLLM_EXPECTED_MODEL=${VLLM_EXPECTED_MODEL:-./configs/vllm_hidden_qwen4b}

echo "========================================"
echo "Online RL with vLLM Hidden States (JAX)"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --list-gpus | head -1)"
echo "========================================"

echo "Environments: $ENVS"
echo "Timesteps: $TIMESTEPS"
echo "Skip-N: $SKIP_N (LLM inference every $SKIP_N steps)"
echo "Num-steps/update: $NUM_STEPS"
echo "Extract layer: $LAYER"
echo "Tokens to generate: $TOKENS"
echo "Fusion mode: $FUSION_MODE"
echo "Hidden gate init logit: $HIDDEN_GATE_INIT_LOGIT"
echo "Actor head layers: $ACTOR_HEAD_LAYERS"
echo "Critic head layers: $CRITIC_HEAD_LAYERS"
echo "No LLM mode: $NO_LLM"
echo "Policy save dir: $POLICY_SAVE_DIR"
echo "Checkpoint every steps: $CHECKPOINT_EVERY_STEPS"
if [[ -n "${CHECKPOINT_DIR}" ]]; then
    echo "Checkpoint dir: $CHECKPOINT_DIR"
fi
if [[ -n "${RESUME_FROM}" ]]; then
    echo "Resume from: $RESUME_FROM"
fi
echo ""

VLLM_PID=""
cleanup() {
    if [[ -n "${VLLM_PID}" ]]; then
        echo "Stopping vLLM server (pid=${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

on_error() {
    local rc=$?
    echo "FATAL: run_online_rl_hidden_jax.sbatch failed (exit=${rc}) at line ${BASH_LINENO[0]} while running: ${BASH_COMMAND}"
    if [[ -n "${VLLM_PID}" ]] && [[ -f "logs/vllm_server_jax_${SLURM_JOB_ID}.log" ]]; then
        echo "Last 80 lines of vLLM server log:"
        tail -80 "logs/vllm_server_jax_${SLURM_JOB_ID}.log" || true
    fi
    exit "${rc}"
}
trap on_error ERR

find_open_port() {
    local port="$1"
    local tries=0
    while [[ $tries -lt 512 ]]; do
        if ! ss -ltn "( sport = :${port} )" 2>/dev/null | grep -q ":${port}"; then
            echo "${port}"
            return 0
        fi
        port=$((port + 1))
        tries=$((tries + 1))
    done
    return 1
}

if [[ "${NO_LLM}" != "1" ]]; then
    JOB_SUFFIX=$((SLURM_JOB_ID % 1000))
    DESIRED_PORT=$((VLLM_PORT_BASE + JOB_SUFFIX))
    VLLM_PORT="$(find_open_port "${DESIRED_PORT}")"
    if [[ -z "${VLLM_PORT}" ]]; then
        echo "❌ ERROR: failed to find a free vLLM port near ${DESIRED_PORT}"
        exit 1
    fi
    export VLLM_PORT
    export VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
    echo "Using VLLM_URL=${VLLM_URL}"

    # Start vLLM server in background
    echo "Starting vLLM server..."
    activate_conda_for "vllm-server" \
        /data/user_data/geney/.conda/envs/craftax_fast_llm \
        craftax_fast_llm \
        base
    bash scripts/start_vllm_hidden.sh --mode last_token --port "${VLLM_PORT}" > logs/vllm_server_jax_${SLURM_JOB_ID}.log 2>&1 &
    VLLM_PID=$!

    # Wait for server to be ready (no timeout - will wait as long as needed)
    echo "Waiting for vLLM server to initialize (this can take several minutes)..."
    echo "The server needs to load all model shards before starting..."
    SECONDS=0
    while true; do
        if curl -fsS "${VLLM_URL}/health" > /dev/null 2>&1 \
           && curl -fsS "${VLLM_URL}/v1/models" | grep -q "${VLLM_EXPECTED_MODEL}"; then
            echo "✅ vLLM server ready after $SECONDS seconds!"
            break
        fi

        # Check if the vLLM process is still running
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "❌ ERROR: vLLM server process died unexpectedly"
            echo "Showing last 50 lines of server log:"
            tail -50 logs/vllm_server_jax_${SLURM_JOB_ID}.log
            exit 1
        fi

        sleep 2
        if [ $((SECONDS % 30)) -eq 0 ] && [ $SECONDS -gt 0 ]; then
            echo "  Still waiting... ($SECONDS seconds elapsed)"
            # Show a snippet of the log to see progress
            echo "  Latest from server log:"
            tail -3 logs/vllm_server_jax_${SLURM_JOB_ID}.log | sed 's/^/    /'
        fi
    done
else
    echo "NO_LLM=1 set: skipping vLLM server startup."
fi

# Switch to craftax environment for JAX training (has distrax)
echo ""
echo "Switching to craftax environment for training..."
activate_conda_for "training" \
    /data/user_data/geney/.conda/envs/craftax \
    craftax \
    base

# Run JAX version
echo "Starting JAX training..."
echo "========================================"
PY_ARGS=(
    --envs "$ENVS"
    --timesteps "$TIMESTEPS"
    --skip-n "$SKIP_N"
    --num-steps "$NUM_STEPS"
    --layer "$LAYER"
    --tokens "$TOKENS"
    --fusion-mode "$FUSION_MODE"
    --hidden-gate-init-logit "$HIDDEN_GATE_INIT_LOGIT"
    --actor-head-layers "$ACTOR_HEAD_LAYERS"
    --critic-head-layers "$CRITIC_HEAD_LAYERS"
    --policy-save-dir "$POLICY_SAVE_DIR"
    --checkpoint-every-steps "$CHECKPOINT_EVERY_STEPS"
    --wandb-project craftax-online-rl-llm
    --wandb-entity iris-sobolmark
)
if [[ -n "${CHECKPOINT_DIR}" ]]; then
    PY_ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi
if [[ -n "${RESUME_FROM}" ]]; then
    PY_ARGS+=(--resume-from "$RESUME_FROM")
fi
if [[ "${NO_LLM}" == "1" ]]; then
    PY_ARGS+=(--no-llm)
fi
if [[ -n "${RUN_NAME}" ]]; then
    PY_ARGS+=(--run-name "$RUN_NAME")
fi
set +e
python -u online_rl_llm/online_rl_hidden_jax.py "${PY_ARGS[@]}"
TRAIN_RC=$?
set -e
if [[ ${TRAIN_RC} -ne 0 ]]; then
    echo "ERROR: training process exited with code ${TRAIN_RC}"
    if [[ -f "logs/vllm_server_jax_${SLURM_JOB_ID}.log" ]]; then
        echo "Last 120 lines of vLLM server log:"
        tail -120 "logs/vllm_server_jax_${SLURM_JOB_ID}.log" || true
    fi
    exit "${TRAIN_RC}"
fi

echo ""
echo "Done! Date: $(date)"
