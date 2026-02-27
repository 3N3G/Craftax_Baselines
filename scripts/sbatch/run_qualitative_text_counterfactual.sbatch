#!/bin/bash
#SBATCH --job-name=qual_text_cf
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/geney/Craftax_Baselines/logs/qual_text_cf_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/qual_text_cf_%j.err

set -Eeuo pipefail

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

MANIFEST="${MANIFEST:-configs/eval/policy_text_counterfactual_5policies.yaml}"
POLICY_IDS="${POLICY_IDS:-}"
BUNDLE_DIR="${BUNDLE_DIR:-golden_examples/game_20260225_211753}"
COUNTERFACTUALS_YAML="${COUNTERFACTUALS_YAML:-golden_examples/game_20260225_211753/manual_counterfactuals_11.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis/qualitative_text_counterfactual}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_STAMP}}"
PREVIEW_ONLY="${PREVIEW_ONLY:-0}"

START_VLLM="${START_VLLM:-1}"
VLLM_SAFE_MODE="${VLLM_SAFE_MODE:-0}"
VLLM_PORT_BASE="${VLLM_PORT_BASE:-21000}"
VLLM_EXPECTED_MODEL="${VLLM_EXPECTED_MODEL:-./configs/vllm_hidden_qwen4b}"
EVAL_ENV_CANDIDATES="${EVAL_ENV_CANDIDATES:-/data/user_data/geney/.conda/envs/imaug,imaug,/data/user_data/geney/.conda/envs/test,test,/data/user_data/geney/.conda/envs/craftax_fast_llm,craftax_fast_llm,/data/user_data/geney/.conda/envs/craftax,craftax,base}"

if [[ "${PREVIEW_ONLY}" == "1" ]]; then
    START_VLLM=0
fi

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
    echo "FATAL: run_qualitative_text_counterfactual.sbatch failed (exit=${rc}) at line ${BASH_LINENO[0]} while running: ${BASH_COMMAND}"
    if [[ -f "logs/vllm_qual_text_cf_${SLURM_JOB_ID}.log" ]]; then
        echo "Last 120 lines of vLLM server log:"
        tail -120 "logs/vllm_qual_text_cf_${SLURM_JOB_ID}.log" || true
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

# Precheck eval env before expensive startup.
PRECHECK_EVAL_ENV_OK=0
PRECHECK_EVAL_ENV=""
PRECHECK_MODS='import importlib
mods = [
    "torch", "jax", "craftax", "flax", "optax", "orbax.checkpoint", "distrax",
    "matplotlib", "requests", "yaml",
]
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
    if ! conda activate /data/user_data/geney/.conda/envs/craftax_fast_llm >/dev/null 2>&1; then
        conda activate craftax_fast_llm >/dev/null 2>&1 || conda activate base >/dev/null 2>&1
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

    if [[ -z "${VLLM_TMP_ROOT:-}" ]]; then
        tmp_candidates=(
            "${SLURM_TMPDIR:-}/craftax_vllm_qual_${SLURM_JOB_ID}"
            "/scratch/${USER}/craftax_vllm_qual_${SLURM_JOB_ID}"
            "/data/user_data/${USER}/scratch/craftax_vllm_qual_${SLURM_JOB_ID}"
            "/tmp/${USER}/craftax_vllm_qual_${SLURM_JOB_ID}"
        )
        for candidate in "${tmp_candidates[@]}"; do
            [[ -z "${candidate}" || "${candidate}" == "/craftax_vllm_qual_${SLURM_JOB_ID}" ]] && continue
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

    bash scripts/start_vllm_hidden.sh "${VLLM_ARGS[@]}" > "logs/vllm_qual_text_cf_${SLURM_JOB_ID}.log" 2>&1 &
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
            tail -100 "logs/vllm_qual_text_cf_${SLURM_JOB_ID}.log" || true
            exit 1
        fi
        if [[ ${SECONDS} -gt 1200 ]]; then
            echo "ERROR: timed out waiting for vLLM server"
            tail -100 "logs/vllm_qual_text_cf_${SLURM_JOB_ID}.log" || true
            exit 1
        fi
        sleep 2
    done
else
    echo "START_VLLM=0: skipping vLLM startup."
fi

activate_eval_env() {
    local py_check='import importlib
mods = [
    "torch", "jax", "craftax", "flax", "optax", "orbax.checkpoint", "distrax",
    "matplotlib", "requests", "yaml",
]
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
            echo "Activated conda env for qualitative eval: ${candidate}"
            return 0
        fi
    done
    echo "ERROR: failed to find eval env with required modules."
    echo "Tried candidates: ${EVAL_ENV_CANDIDATES}"
    conda env list || true
    return 1
}

activate_eval_env

mkdir -p "${OUTPUT_DIR}"

CMD=(
    python -u scripts/qualitative_text_counterfactual.py
    --manifest "${MANIFEST}"
    --bundle_dir "${BUNDLE_DIR}"
    --counterfactuals_yaml "${COUNTERFACTUALS_YAML}"
    --output_dir "${OUTPUT_DIR}"
)
if [[ -n "${POLICY_IDS}" ]]; then
    CMD+=(--policy_ids "${POLICY_IDS}")
fi
if [[ "${PREVIEW_ONLY}" == "1" ]]; then
    CMD+=(--preview_only)
fi

printf '  %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "Qualitative counterfactual outputs written to: ${OUTPUT_DIR}"
