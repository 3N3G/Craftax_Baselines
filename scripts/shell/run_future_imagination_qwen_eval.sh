#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <model_id> <model_tag> <port>"
  echo "Example: $0 Qwen/Qwen3-4B qwen3-4b 8010"
  exit 2
fi

MODEL_ID="$1"
MODEL_TAG="$2"
PORT="$3"

TRAJ_DIR="play_data/trajectory_records/traj_20260311_100410"
STATE_CFG="configs/future_imagination/run_config_predict_stateonly_key7_traj_20260311_100410.json"
HISTORY_CFG="configs/future_imagination/run_config_predict_historyk5_key7_traj_20260311_100410.json"
BASE_OUT="analysis/future_imagination"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"

MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
READY_TIMEOUT_S="${VLLM_READY_TIMEOUT_S:-3600}"
READY_POLL_S="${VLLM_READY_POLL_S:-5}"
INSTALL_TRANSFORMERS_MAIN="${INSTALL_TRANSFORMERS_MAIN:-0}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
STATE_OUT="${BASE_OUT}/${RUN_TS}_traj_20260311_100410_predict_stateonly_key7_promptrev_${MODEL_TAG}"
HIST_OUT="${BASE_OUT}/${RUN_TS}_traj_20260311_100410_predict_historyk5_key7_promptrev_${MODEL_TAG}"

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  TMP_ROOT="${SLURM_TMPDIR}/future_imagination_${MODEL_TAG}_${SLURM_JOB_ID:-nojob}"
else
  TMP_ROOT="/scratch/${USER}/future_imagination_${MODEL_TAG}_${SLURM_JOB_ID:-nojob}"
fi
mkdir -p "${TMP_ROOT}"
export TMPDIR="${TMP_ROOT}"
export VLLM_CACHE_ROOT="${TMP_ROOT}/vllm_cache"
export HF_HOME="${TMP_ROOT}/hf_home"
export HUGGINGFACE_HUB_CACHE="${TMP_ROOT}/hf_home/hub"
export TRANSFORMERS_CACHE="${TMP_ROOT}/hf_home/transformers"
mkdir -p "${VLLM_CACHE_ROOT}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[info] model=${MODEL_ID} tag=${MODEL_TAG} port=${PORT}"
echo "[info] state_out=${STATE_OUT}"
echo "[info] history_out=${HIST_OUT}"
echo "[info] ready_timeout_s=${READY_TIMEOUT_S} ready_poll_s=${READY_POLL_S}"
echo "[info] install_transformers_main=${INSTALL_TRANSFORMERS_MAIN}"
echo "[info] vllm_extra_args=${VLLM_EXTRA_ARGS}"

if [[ "${INSTALL_TRANSFORMERS_MAIN}" == "1" ]]; then
  PY_OVERLAY="${TMP_ROOT}/py_overlay"
  mkdir -p "${PY_OVERLAY}"
  echo "[info] installing transformers(main) overlay into ${PY_OVERLAY}"
  python3 -m pip install --quiet --target "${PY_OVERLAY}" --upgrade \
    "huggingface_hub>=0.34.0" \
    "tokenizers>=0.22.0"
  python3 -m pip install --quiet --target "${PY_OVERLAY}" --upgrade \
    "git+https://github.com/huggingface/transformers.git"
  export PYTHONPATH="${PY_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
  python3 - <<'PY'
import importlib
from transformers import AutoConfig
print("transformers_overlay_version", importlib.import_module("transformers").__version__)
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B")
print("qwen35_config_ok", cfg.model_type)
PY
fi

python3 - <<'PY'
import importlib
mods = ["numpy", "requests", "vllm"]
for m in mods:
    importlib.import_module(m)
print("python_preflight_ok")
PY

VLLM_LOG="${BASE_OUT}/vllm_${MODEL_TAG}_${RUN_TS}.log"
mkdir -p "${BASE_OUT}"

EXTRA_ARGS=()
if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${VLLM_EXTRA_ARGS}"
fi

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

vllm serve "${MODEL_ID}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}" \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

echo "[info] vllm_pid=${VLLM_PID} log=${VLLM_LOG}"
READY_TRIES=$((READY_TIMEOUT_S / READY_POLL_S))
if [[ "${READY_TRIES}" -le 0 ]]; then
  READY_TRIES=1
fi
for i in $(seq 1 "${READY_TRIES}"); do
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "[error] vllm process exited before readiness check passed"
    tail -n 120 "${VLLM_LOG}" || true
    exit 1
  fi
  if curl -sSf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[info] vllm_ready"
    break
  fi
  if (( i % 12 == 0 )); then
    echo "[info] waiting_for_vllm_ready elapsed_s=$((i * READY_POLL_S))"
  fi
  sleep "${READY_POLL_S}"
done

if ! curl -sSf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[error] vllm did not become ready in time"
  tail -n 200 "${VLLM_LOG}" || true
  exit 1
fi

python3 scripts/future_imagination_eval.py \
  --trajectory-dir "${TRAJ_DIR}" \
  --config "${STATE_CFG}" \
  --provider openai_compatible \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-dir "${STATE_OUT}" \
  --store-full-prompts \
  --resume \
  --request-timeout-s 300

python3 scripts/future_imagination_eval.py \
  --trajectory-dir "${TRAJ_DIR}" \
  --config "${HISTORY_CFG}" \
  --provider openai_compatible \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-dir "${HIST_OUT}" \
  --store-full-prompts \
  --resume \
  --request-timeout-s 300

echo "[done] state_out=${STATE_OUT}"
echo "[done] history_out=${HIST_OUT}"
