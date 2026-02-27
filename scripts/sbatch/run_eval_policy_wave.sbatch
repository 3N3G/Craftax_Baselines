#!/bin/bash
#SBATCH --job-name=eval_policy_wave
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/geney/Craftax_Baselines/logs/eval_policy_wave_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/eval_policy_wave_%j.err

set -Eeuo pipefail

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

MANIFEST="${MANIFEST:-configs/eval/policy_wave_v2.yaml}"
TRACKS="${TRACKS:-id,value,ood,gameplay_llm}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/policy_wave_v2}"
SEEDS="${SEEDS:-42,43,44}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EPISODES="${NUM_EPISODES:-128}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-80000}"
INCLUDE_SLICES="${INCLUDE_SLICES:-1}"
SLICE_COUNT="${SLICE_COUNT:-3}"
DETERMINISTIC="${DETERMINISTIC:-0}"

LLM_MODEL_ID="${LLM_MODEL_ID:-Qwen/Qwen3-4B}"
LLM_LAYER="${LLM_LAYER:--1}"
LLM_HIDDEN_TOKENS="${LLM_HIDDEN_TOKENS:-1}"
LLM_GENERATION_TOKENS="${LLM_GENERATION_TOKENS:-16}"

VALUE_DATASET_DIR="${VALUE_DATASET_DIR:-/data/group_data/rl/geney/vllm_craftax_labelled_results}"
VALUE_DATA_GLOB="${VALUE_DATA_GLOB:-trajectories_batch_*.npz}"
VALUE_NUM_SAMPLES="${VALUE_NUM_SAMPLES:-20000}"
VALUE_HIDDEN_MODE="${VALUE_HIDDEN_MODE:-real}"
VALUE_PAIRS="${VALUE_PAIRS:-health,food,drink,energy,wood,stone,floor1_zombie_adjacent,floor1_skeleton_adjacent,floor2_witch_adjacent}"
BUNDLE_DIRS="${BUNDLE_DIRS:-}"

NO_WANDB="${NO_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-craftax_symbolic_evals}"
WANDB_ENTITY="${WANDB_ENTITY:-iris-sobolmark}"

START_VLLM="${START_VLLM:-1}"
VLLM_SAFE_MODE="${VLLM_SAFE_MODE:-0}"
VLLM_PORT_BASE="${VLLM_PORT_BASE:-20000}"
VLLM_EXPECTED_MODEL="${VLLM_EXPECTED_MODEL:-./configs/vllm_hidden_qwen4b}"
EVAL_ENV_CANDIDATES="${EVAL_ENV_CANDIDATES:-/data/user_data/geney/.conda/envs/imaug,imaug,/data/user_data/geney/.conda/envs/test,test,/data/user_data/geney/.conda/envs/craftax_fast_llm,craftax_fast_llm,/data/user_data/geney/.conda/envs/craftax,craftax,base}"

POLICY_IDS="${POLICY_IDS:-}"
SUMMARY_PATH="${SUMMARY_PATH:-}"

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
    echo "FATAL: run_eval_policy_wave.sbatch failed (exit=${rc}) at line ${BASH_LINENO[0]} while running: ${BASH_COMMAND}"
    if [[ -f "logs/vllm_eval_policy_wave_${SLURM_JOB_ID}.log" ]]; then
        echo "Last 120 lines of vLLM eval server log:"
        tail -120 "logs/vllm_eval_policy_wave_${SLURM_JOB_ID}.log" || true
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

# Precheck eval env before spending minutes on vLLM startup.
PRECHECK_EVAL_ENV_OK=0
PRECHECK_EVAL_ENV=""
PRECHECK_MODS='import importlib
mods = ["torch", "jax", "craftax", "flax", "optax", "orbax.checkpoint", "distrax"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m}: {exc.__class__.__name__}: {exc}")
if missing:
    raise SystemExit("\\n".join(missing))
print("python_import_check_ok")
'
IFS=',' read -r -a _precheck_candidates <<< "${EVAL_ENV_CANDIDATES}"
for _raw in "${_precheck_candidates[@]}"; do
    _candidate="$(echo "${_raw}" | xargs)"
    [[ -z "${_candidate}" ]] && continue
    if ! conda activate "${_candidate}" >/dev/null 2>&1; then
        continue
    fi
    if python - <<PY >/dev/null 2>&1
${PRECHECK_MODS}
PY
    then
        PRECHECK_EVAL_ENV_OK=1
        PRECHECK_EVAL_ENV="${_candidate}"
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

if [[ "${START_VLLM}" == "1" ]]; then
    activate_conda_for "vllm-server" \
        /data/user_data/geney/.conda/envs/craftax_fast_llm \
        craftax_fast_llm \
        base

    JOB_SUFFIX=$((SLURM_JOB_ID % 1000))
    # Reserve spacing between jobs because vLLM may probe nearby ports for internal rendezvous.
    DESIRED_PORT=$((VLLM_PORT_BASE + JOB_SUFFIX * 20))
    VLLM_PORT="$(find_open_port "${DESIRED_PORT}")"
    if [[ -z "${VLLM_PORT}" ]]; then
        echo "ERROR: failed to find free vLLM port near ${DESIRED_PORT}"
        exit 1
    fi
    export VLLM_PORT
    export VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
    if [[ -z "${VLLM_TMP_ROOT:-}" ]]; then
        tmp_candidates=(
            "${SLURM_TMPDIR:-}/craftax_vllm_eval_${SLURM_JOB_ID}"
            "/scratch/${USER}/craftax_vllm_eval_${SLURM_JOB_ID}"
            "/data/user_data/${USER}/scratch/craftax_vllm_eval_${SLURM_JOB_ID}"
            "/tmp/${USER}/craftax_vllm_eval_${SLURM_JOB_ID}"
        )
        for candidate in "${tmp_candidates[@]}"; do
            [[ -z "${candidate}" || "${candidate}" == "/craftax_vllm_eval_${SLURM_JOB_ID}" ]] && continue
            if mkdir -p "${candidate}" >/dev/null 2>&1; then
                export VLLM_TMP_ROOT="${candidate}"
                break
            fi
        done
        if [[ -z "${VLLM_TMP_ROOT:-}" ]]; then
            echo "ERROR: unable to create VLLM_TMP_ROOT in any candidate location"
            printf 'Candidates tried:\n'
            printf '  %s\n' "${tmp_candidates[@]}"
            exit 1
        fi
    else
        mkdir -p "${VLLM_TMP_ROOT}"
    fi
    echo "Using VLLM_URL=${VLLM_URL}"
    echo "Using VLLM_TMP_ROOT=${VLLM_TMP_ROOT}"

    VLLM_ARGS=(--mode last_token --port "${VLLM_PORT}")
    if [[ "${VLLM_SAFE_MODE}" == "1" ]]; then
        VLLM_ARGS+=(--safe)
    fi

    bash scripts/start_vllm_hidden.sh "${VLLM_ARGS[@]}" > "logs/vllm_eval_policy_wave_${SLURM_JOB_ID}.log" 2>&1 &
    VLLM_PID=$!

    echo "Waiting for vLLM /health and /v1/models..."
    SECONDS=0
    while true; do
        if curl -fsS "${VLLM_URL}/health" >/dev/null 2>&1 \
           && curl -fsS "${VLLM_URL}/v1/models" | grep -q "${VLLM_EXPECTED_MODEL}"; then
            echo "vLLM ready after ${SECONDS}s"
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM server process died"
            tail -100 "logs/vllm_eval_policy_wave_${SLURM_JOB_ID}.log" || true
            exit 1
        fi
        if [[ ${SECONDS} -gt 1200 ]]; then
            echo "ERROR: timed out waiting for vLLM server"
            tail -100 "logs/vllm_eval_policy_wave_${SLURM_JOB_ID}.log" || true
            exit 1
        fi
        sleep 2
    done
else
    echo "START_VLLM=0: skipping vLLM startup."
fi

activate_eval_env() {
    local py_check='import importlib
mods = ["torch", "jax", "craftax", "flax", "optax", "orbax.checkpoint", "distrax"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m}: {exc.__class__.__name__}: {exc}")
if missing:
    raise SystemExit("\\n".join(missing))
print("python_import_check_ok")
'
    local raw candidate
    IFS=',' read -r -a candidates <<< "${EVAL_ENV_CANDIDATES}"
    for raw in "${candidates[@]}"; do
        candidate="$(echo "${raw}" | xargs)"
        [[ -z "${candidate}" ]] && continue
        if ! conda activate "${candidate}" >/dev/null 2>&1; then
            continue
        fi
        if python - <<PY >/dev/null 2>&1
${py_check}
PY
        then
            echo "Activated conda env for policy-wave-eval: ${candidate}"
            return 0
        fi
    done
    echo "ERROR: failed to find policy eval env with required modules."
    echo "Tried candidates: ${EVAL_ENV_CANDIDATES}"
    conda env list || true
    return 1
}

activate_eval_env

CMD=(
    python -u scripts/eval_policy_wave.py
    --manifest "${MANIFEST}"
    --tracks "${TRACKS}"
    --output_dir "${OUTPUT_DIR}"
    --seeds "${SEEDS}"
    --num_envs "${NUM_ENVS}"
    --num_episodes "${NUM_EPISODES}"
    --max_env_steps "${MAX_ENV_STEPS}"
    --slice_count "${SLICE_COUNT}"
    --llm_model_id "${LLM_MODEL_ID}"
    --llm_layer "${LLM_LAYER}"
    --llm_hidden_tokens "${LLM_HIDDEN_TOKENS}"
    --llm_generation_tokens "${LLM_GENERATION_TOKENS}"
    --value_dataset_dir "${VALUE_DATASET_DIR}"
    --value_data_glob "${VALUE_DATA_GLOB}"
    --value_num_samples "${VALUE_NUM_SAMPLES}"
    --value_hidden_mode "${VALUE_HIDDEN_MODE}"
    --value_pairs "${VALUE_PAIRS}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_entity "${WANDB_ENTITY}"
)

if [[ "${INCLUDE_SLICES}" == "1" ]]; then
    CMD+=(--include_slices)
else
    CMD+=(--no-include_slices)
fi

if [[ "${DETERMINISTIC}" == "1" ]]; then
    CMD+=(--deterministic)
fi

if [[ -n "${BUNDLE_DIRS}" ]]; then
    CMD+=(--bundle_dirs "${BUNDLE_DIRS}")
fi

if [[ -n "${POLICY_IDS}" ]]; then
    CMD+=(--policy_ids "${POLICY_IDS}")
fi

if [[ -n "${SUMMARY_PATH}" ]]; then
    CMD+=(--summary_path "${SUMMARY_PATH}")
fi

if [[ "${NO_WANDB}" == "1" ]]; then
    CMD+=(--no_wandb)
fi

echo "Running eval wave command:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Done: $(date)"
