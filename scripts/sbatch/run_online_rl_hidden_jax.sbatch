#!/bin/bash
#SBATCH --job-name=online_rl_hidden_jax
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=logs/online_rl_hidden_jax_%j.out
#SBATCH --error=logs/online_rl_hidden_jax_%j.err

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

python_import_preflight() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 180 python - <<'PY' >/dev/null
import importlib
mods = [
    "jax",
    "flax",
    "optax",
    "distrax",
    "craftax",
    "requests",
    "numpy",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m}: {exc.__class__.__name__}: {exc}")
if missing:
    raise SystemExit("\n".join(missing))
print("python_import_check_ok")
PY
    else
        python - <<'PY' >/dev/null
import importlib
mods = [
    "jax",
    "flax",
    "optax",
    "distrax",
    "craftax",
    "requests",
    "numpy",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m}: {exc.__class__.__name__}: {exc}")
if missing:
    raise SystemExit("\n".join(missing))
print("python_import_check_ok")
PY
    fi
}

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

resolve_resume_step() {
    local resume_path="$1"
    if [[ -z "${resume_path}" ]]; then
        echo 0
        return 0
    fi
    python - "$resume_path" <<'PY'
import json
import os
import re
import sys

path = sys.argv[1]

def extract_step(p):
    m = re.search(r"_resume_step(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1

if os.path.isfile(path):
    print(max(0, extract_step(path)))
    raise SystemExit(0)
if os.path.isdir(path):
    latest = os.path.join(path, "latest_resume.json")
    if os.path.exists(latest):
        try:
            with open(latest, "r", encoding="utf-8") as f:
                payload = json.load(f)
            print(max(0, int(payload.get("total_steps", 0))))
            raise SystemExit(0)
        except Exception:
            pass
    candidates = []
    for name in os.listdir(path):
        if name.endswith(".pkl") and "_resume_step" in name:
            candidates.append(extract_step(name))
    print(max([0] + candidates))
    raise SystemExit(0)
print(0)
PY
}

latest_checkpoint_step() {
    local ckpt_dir="$1"
    if [[ -z "${ckpt_dir}" ]]; then
        echo 0
        return 0
    fi
    python - "$ckpt_dir" <<'PY'
import json
import os
import re
import sys

ckpt_dir = sys.argv[1]
if not os.path.isdir(ckpt_dir):
    print(0)
    raise SystemExit(0)

latest_path = os.path.join(ckpt_dir, "latest_resume.json")
if os.path.exists(latest_path):
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        print(int(payload.get("total_steps", 0)))
        raise SystemExit(0)
    except Exception:
        pass

step_re = re.compile(r"_resume_step(\d+)")
best = 0
for name in os.listdir(ckpt_dir):
    if not name.endswith(".pkl"):
        continue
    m = step_re.search(name)
    if m:
        best = max(best, int(m.group(1)))
print(best)
PY
}

choose_vllm_tmp_root() {
    if [[ -n "${VLLM_TMP_ROOT:-}" ]]; then
        mkdir -p "${VLLM_TMP_ROOT}"
        echo "${VLLM_TMP_ROOT}"
        return 0
    fi

    local candidates=(
        "${SLURM_TMPDIR:-}/craftax_vllm_${SLURM_JOB_ID}"
        "/scratch/${USER}/craftax_vllm_${SLURM_JOB_ID}"
        "/data/user_data/${USER}/scratch/craftax_vllm_${SLURM_JOB_ID}"
        "/tmp/${USER}/craftax_vllm_${SLURM_JOB_ID}"
    )
    local c
    for c in "${candidates[@]}"; do
        [[ -z "${c}" || "${c}" == "/craftax_vllm_${SLURM_JOB_ID}" ]] && continue
        if mkdir -p "${c}" >/dev/null 2>&1; then
            echo "${c}"
            return 0
        fi
    done
    return 1
}

# Parameters
ENVS=${1:-8}
TIMESTEPS=${2:-1000000}
SKIP_N=${3:-1}
LAYER=${4:--1}
TOKENS=${5:-1}
NUM_STEPS=${6:-64}
CHECKPOINT_EVERY_STEPS=${7:-10000000}
POLICY_SAVE_DIR=${8:-/data/group_data/rl/geney/online_rl_hidden_models}
CHECKPOINT_DIR=${9:-}
RESUME_FROM=${10:-}
NO_LLM=${NO_LLM:-0}
RUN_NAME=${RUN_NAME:-}
VLLM_PORT_BASE=${VLLM_PORT_BASE:-18000}
VLLM_EXPECTED_MODEL=${VLLM_EXPECTED_MODEL:-./configs/vllm_hidden_qwen4b}
HIDDEN_POOLING=${HIDDEN_POOLING:-last_token}
HIDDEN_POOLING_K=${HIDDEN_POOLING_K:-8}
TEMPERATURE=${TEMPERATURE:-0.7}
SAVE_TRAJ_ONLINE=${SAVE_TRAJ_ONLINE:-0}
TRAJ_SAVE_DIR=${TRAJ_SAVE_DIR:-}
TRAJ_SAVE_EVERY_UPDATES=${TRAJ_SAVE_EVERY_UPDATES:-50}
TRAJ_FREE_SPACE_MIN_GB=${TRAJ_FREE_SPACE_MIN_GB:-150}
TRAJ_SCHEMA=${TRAJ_SCHEMA:-minimal_core}

CHAIN_ENABLED=${CHAIN_ENABLED:-0}
CHAIN_TARGET_STEPS=${CHAIN_TARGET_STEPS:-${TIMESTEPS}}
CHAIN_CHECKPOINT_DIR=${CHAIN_CHECKPOINT_DIR:-}
CHAIN_RUN_NAME=${CHAIN_RUN_NAME:-${RUN_NAME}}

EFFECTIVE_CHECKPOINT_DIR="${CHECKPOINT_DIR:-${POLICY_SAVE_DIR}}"
if [[ -z "${CHAIN_CHECKPOINT_DIR}" ]]; then
    CHAIN_CHECKPOINT_DIR="${EFFECTIVE_CHECKPOINT_DIR}"
fi
CHAIN_START_STEP="$(resolve_resume_step "${RESUME_FROM}")"

echo "========================================"
echo "Online RL with vLLM Hidden States (JAX)"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --list-gpus | head -1)"
echo "========================================"

echo "Environments: $ENVS"
echo "Timesteps target: $TIMESTEPS"
echo "Skip-N: $SKIP_N (LLM inference every $SKIP_N steps)"
echo "Num-steps/update: $NUM_STEPS"
echo "Extract layer: $LAYER"
echo "Tokens to generate: $TOKENS"
echo "Hidden pooling: ${HIDDEN_POOLING} (k=${HIDDEN_POOLING_K})"
echo "Temperature: ${TEMPERATURE}"
echo "No LLM mode: $NO_LLM"
echo "Policy save dir: $POLICY_SAVE_DIR"
echo "Checkpoint every steps: $CHECKPOINT_EVERY_STEPS"
echo "Effective checkpoint dir: ${EFFECTIVE_CHECKPOINT_DIR}"
echo "Resume from: ${RESUME_FROM:-<none>}"
echo "Trajectory save enabled: ${SAVE_TRAJ_ONLINE}"
if [[ "${SAVE_TRAJ_ONLINE}" == "1" ]]; then
    echo "Trajectory dir: ${TRAJ_SAVE_DIR:-<auto>}"
    echo "Trajectory save every updates: ${TRAJ_SAVE_EVERY_UPDATES}"
    echo "Trajectory free-space floor: ${TRAJ_FREE_SPACE_MIN_GB} GB"
fi
echo "Chain enabled: ${CHAIN_ENABLED}"
echo "Chain start step: ${CHAIN_START_STEP}"
echo "Chain target steps: ${CHAIN_TARGET_STEPS}"
echo "Chain checkpoint dir: ${CHAIN_CHECKPOINT_DIR}"
echo ""

VLLM_PID=""
TRAIN_PID=""
TRAIN_RC=1
TERM_REQUESTED=0

cleanup() {
    if [[ -n "${VLLM_PID}" ]]; then
        echo "Stopping vLLM server (pid=${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

on_term_signal() {
    TERM_REQUESTED=1
    echo "Received termination signal; requesting graceful checkpoint from python process."
    if [[ -n "${TRAIN_PID}" ]]; then
        kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    fi
}
trap on_term_signal TERM INT

if [[ "${NO_LLM}" != "1" ]]; then
    # Preflight training env before expensive vLLM startup.
    activate_conda_for "training-preflight" \
        /data/user_data/geney/.conda/envs/craftax \
        craftax \
        base
    if ! python_import_preflight; then
        echo "ERROR: training env import preflight failed."
        exit 1
    fi

    JOB_SUFFIX=$((SLURM_JOB_ID % 1000))
    DESIRED_PORT=$((VLLM_PORT_BASE + JOB_SUFFIX * 20))
    VLLM_PORT="$(find_open_port "${DESIRED_PORT}")"
    if [[ -z "${VLLM_PORT}" ]]; then
        echo "ERROR: failed to find a free vLLM port near ${DESIRED_PORT}"
        exit 1
    fi
    export VLLM_PORT
    export VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
    VLLM_TMP_ROOT="$(choose_vllm_tmp_root)" || {
        echo "ERROR: unable to select VLLM_TMP_ROOT from safe candidates"
        exit 1
    }
    export VLLM_TMP_ROOT
    echo "Using VLLM_URL=${VLLM_URL}"
    echo "Using VLLM_TMP_ROOT=${VLLM_TMP_ROOT}"

    VLLM_MODE="last_token"
    if [[ "${HIDDEN_POOLING}" == "mean_last_k" ]]; then
        VLLM_MODE="all"
    fi

    echo "Starting vLLM server (mode=${VLLM_MODE})..."
    activate_conda_for "vllm-server" \
        /data/user_data/geney/.conda/envs/craftax_fast_llm \
        craftax_fast_llm \
        base
    bash scripts/start_vllm_hidden.sh \
        --mode "${VLLM_MODE}" \
        --port "${VLLM_PORT}" > "logs/vllm_server_jax_${SLURM_JOB_ID}.log" 2>&1 &
    VLLM_PID=$!

    echo "Waiting for vLLM server to initialize..."
    SECONDS=0
    while true; do
        if curl -fsS "${VLLM_URL}/health" >/dev/null 2>&1 \
           && curl -fsS "${VLLM_URL}/v1/models" | grep -q "${VLLM_EXPECTED_MODEL}"; then
            echo "vLLM server ready after ${SECONDS}s"
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM server process died unexpectedly"
            tail -80 "logs/vllm_server_jax_${SLURM_JOB_ID}.log" || true
            exit 1
        fi
        sleep 2
        if [[ ${SECONDS} -gt 1200 ]]; then
            echo "ERROR: timed out waiting for vLLM server"
            tail -120 "logs/vllm_server_jax_${SLURM_JOB_ID}.log" || true
            exit 1
        fi
        if [[ $((SECONDS % 30)) -eq 0 ]] && [[ ${SECONDS} -gt 0 ]]; then
            echo "  Still waiting... (${SECONDS}s)"
            tail -3 "logs/vllm_server_jax_${SLURM_JOB_ID}.log" | sed 's/^/    /'
        fi
    done
else
    activate_conda_for "training-preflight" \
        /data/user_data/geney/.conda/envs/craftax \
        craftax \
        base
    if ! python_import_preflight; then
        echo "ERROR: training env import preflight failed."
        exit 1
    fi
    echo "NO_LLM=1 set: skipping vLLM startup."
fi

# Switch to training env and run.
activate_conda_for "training" \
    /data/user_data/geney/.conda/envs/craftax \
    craftax \
    base

PY_ARGS=(
    --envs "$ENVS"
    --timesteps "$TIMESTEPS"
    --skip-n "$SKIP_N"
    --num-steps "$NUM_STEPS"
    --layer "$LAYER"
    --tokens "$TOKENS"
    --hidden-pooling "$HIDDEN_POOLING"
    --hidden-pooling-k "$HIDDEN_POOLING_K"
    --temperature "$TEMPERATURE"
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
if [[ "${SAVE_TRAJ_ONLINE}" == "1" ]]; then
    PY_ARGS+=(--save-traj-online)
    if [[ -n "${TRAJ_SAVE_DIR}" ]]; then
        PY_ARGS+=(--traj-save-dir "$TRAJ_SAVE_DIR")
    fi
    PY_ARGS+=(--traj-save-every-updates "$TRAJ_SAVE_EVERY_UPDATES")
    PY_ARGS+=(--traj-free-space-min-gb "$TRAJ_FREE_SPACE_MIN_GB")
    PY_ARGS+=(--traj-schema "$TRAJ_SCHEMA")
fi

echo "Launching training..."
set +e
python -u online_rl_llm/online_rl_hidden_jax.py "${PY_ARGS[@]}" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
TRAIN_RC=$?
set -e
echo "Training exit code: ${TRAIN_RC}"

if [[ ${TRAIN_RC} -ne 0 ]]; then
    echo "ERROR: training process exited with non-zero status."
    if [[ -f "logs/vllm_server_jax_${SLURM_JOB_ID}.log" ]]; then
        tail -120 "logs/vllm_server_jax_${SLURM_JOB_ID}.log" || true
    fi
fi

maybe_submit_successor() {
    if [[ "${CHAIN_ENABLED}" != "1" ]]; then
        return 0
    fi

    local latest_step
    latest_step="$(latest_checkpoint_step "${CHAIN_CHECKPOINT_DIR}")"
    echo "Chain checkpoint gate: start=${CHAIN_START_STEP} latest=${latest_step} target=${CHAIN_TARGET_STEPS}"

    if [[ -z "${latest_step}" ]]; then
        echo "CHAIN: no latest checkpoint step detected; stopping chain."
        return 0
    fi
    if (( latest_step <= CHAIN_START_STEP )); then
        echo "CHAIN: no checkpoint progress in this segment; stopping chain."
        return 0
    fi
    if (( latest_step >= CHAIN_TARGET_STEPS )); then
        echo "CHAIN: target reached (${latest_step} >= ${CHAIN_TARGET_STEPS}); no successor."
        return 0
    fi

    local sbatch_file="scripts/sbatch/run_online_rl_hidden_jax.sbatch"
    local next_export
    next_export=$(
        printf "ALL,RUN_NAME=%s,CHECKPOINT_DIR=%s,RESUME_FROM=%s,CHAIN_ENABLED=1,CHAIN_TARGET_STEPS=%s,CHAIN_CHECKPOINT_DIR=%s,CHAIN_RUN_NAME=%s" \
            "${CHAIN_RUN_NAME}" \
            "${CHAIN_CHECKPOINT_DIR}" \
            "${CHAIN_CHECKPOINT_DIR}" \
            "${CHAIN_TARGET_STEPS}" \
            "${CHAIN_CHECKPOINT_DIR}" \
            "${CHAIN_RUN_NAME}"
    )
    local submit_out
    submit_out=$(
        sbatch \
            --dependency="afterany:${SLURM_JOB_ID}" \
            --job-name "${SLURM_JOB_NAME}" \
            --export "${next_export}" \
            "${sbatch_file}" \
            "${ENVS}" \
            "${TIMESTEPS}" \
            "${SKIP_N}" \
            "${LAYER}" \
            "${TOKENS}" \
            "${NUM_STEPS}" \
            "${CHECKPOINT_EVERY_STEPS}" \
            "${POLICY_SAVE_DIR}"
    )
    echo "CHAIN: submitted successor => ${submit_out}"
}

maybe_submit_successor

echo "Done! Date: $(date)"
exit "${TRAIN_RC}"
