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
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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
VLLM_PORT_BASE=${VLLM_PORT_BASE:-20000}
MANIFEST=${MANIFEST:-}
MANIFEST_POLICY_IDS=${MANIFEST_POLICY_IDS:-}
POLICIES=${POLICIES:-}
EVAL_ENV_CANDIDATES=${EVAL_ENV_CANDIDATES:-/data/user_data/geney/.conda/envs/imaug,imaug,/data/user_data/geney/.conda/envs/craftax,craftax,base}

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

# Precheck eval env before spending GPU time on vLLM startup.
PRECHECK_EVAL_ENV_OK=0
PRECHECK_EVAL_ENV=""
PRECHECK_EVAL_PY=""
REQUIRE_GPU_JAX=${REQUIRE_GPU_JAX:-1}
PRECHECK_MODS='import importlib
import os
mods = ["torch", "jax", "craftax", "flax", "optax", "orbax.checkpoint", "distrax", "cv2"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m}: {exc.__class__.__name__}: {exc}")
if missing:
    raise SystemExit("\\n".join(missing))
require_gpu = os.environ.get("REQUIRE_GPU_JAX", "1") == "1"
if require_gpu:
    import jax
    backend = jax.default_backend()
    if backend != "gpu":
        raise SystemExit(f"jax_backend_not_gpu: {backend}")
print("python_import_check_ok")
'
IFS=',' read -r -a _precheck_candidates <<< "${EVAL_ENV_CANDIDATES}"
for _raw in "${_precheck_candidates[@]}"; do
    _candidate="$(echo "${_raw}" | xargs)"
    [[ -z "${_candidate}" ]] && continue
    _py_candidate=""
    if [[ -x "${_candidate}/bin/python" ]]; then
        _py_candidate="${_candidate}/bin/python"
    elif conda activate "${_candidate}" >/dev/null 2>&1; then
        _py_candidate="$(command -v python || true)"
    fi
    [[ -z "${_py_candidate}" ]] && continue
    if "${_py_candidate}" - <<PY >/dev/null 2>&1
${PRECHECK_MODS}
PY
    then
        PRECHECK_EVAL_ENV_OK=1
        PRECHECK_EVAL_ENV="${_candidate}"
        PRECHECK_EVAL_PY="${_py_candidate}"
        break
    fi
done
if [[ ${PRECHECK_EVAL_ENV_OK} -ne 1 ]]; then
    echo "ERROR: no usable eval env found during precheck."
    echo "Tried candidates: ${EVAL_ENV_CANDIDATES}"
    conda env list || true
    exit 1
fi
echo "Precheck selected eval env: ${PRECHECK_EVAL_ENV}"
echo "Precheck selected eval python: ${PRECHECK_EVAL_PY}"

if [[ "${START_VLLM}" == "1" ]]; then
    echo "Starting vLLM hidden-state server..."
    VLLM_BIN_DIR="/data/user_data/geney/.conda/envs/craftax_fast_llm/bin"
    if [[ ! -x "${VLLM_BIN_DIR}/python" ]]; then
        activate_conda_for "vllm-server" \
            /data/user_data/geney/.conda/envs/craftax_fast_llm \
            craftax_fast_llm \
            base
        VLLM_BIN_DIR="$(dirname "$(command -v python)")"
        conda deactivate >/dev/null 2>&1 || true
    fi

    JOB_SUFFIX=$((SLURM_JOB_ID % 1000))
    DESIRED_PORT=$((VLLM_PORT_BASE + JOB_SUFFIX * 20))
    VLLM_PORT="$(find_open_port "${DESIRED_PORT}")"
    if [[ -z "${VLLM_PORT}" ]]; then
        echo "ERROR: failed to find free vLLM port near ${DESIRED_PORT}"
        exit 1
    fi
    export VLLM_PORT
    export VLLM_URL="http://127.0.0.1:${VLLM_PORT}"

    VLLM_ARGS=(--mode last_token)
    if [[ "${VLLM_SAFE_MODE}" == "1" ]]; then
        VLLM_ARGS+=(--safe)
    fi
    VLLM_ARGS+=(--port "${VLLM_PORT}")
    PATH="${VLLM_BIN_DIR}:${PATH}" bash scripts/start_vllm_hidden.sh "${VLLM_ARGS[@]}" > "logs/vllm_symbolic_eval_${SLURM_JOB_ID}.log" 2>&1 &
    VLLM_PID=$!

    echo "Waiting for vLLM /health at ${VLLM_URL}..."
    SECONDS=0
    while true; do
        if curl -fsS "${VLLM_URL}/health" >/dev/null 2>&1; then
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
EVAL_PY="${PRECHECK_EVAL_PY}"
if [[ -z "${EVAL_PY}" || ! -x "${EVAL_PY}" ]]; then
    activate_conda_for "policy-eval" "${PRECHECK_EVAL_ENV}"
    EVAL_PY="$(command -v python)"
fi

EXTRA_ARGS=()
if [[ -n "${MANIFEST}" ]]; then
    EXTRA_ARGS+=(--manifest "${MANIFEST}")
fi
if [[ -n "${MANIFEST_POLICY_IDS}" ]]; then
    EXTRA_ARGS+=(--manifest-policy-ids "${MANIFEST_POLICY_IDS}")
fi
if [[ -n "${POLICIES}" ]]; then
    EXTRA_ARGS+=(--policies "${POLICIES}")
fi

echo "Running evaluation suite..."
"${EVAL_PY}" -u scripts/eval_symbolic_policy_suite.py \
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
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "Done: $(date)"
echo "Output JSON: ${OUTPUT_JSON}"
