#!/bin/bash
#SBATCH --job-name=latbench_plugin_fast
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=logs/latbench_plugin_fast_%j.out
#SBATCH --error=logs/latbench_plugin_fast_%j.err

set -eo pipefail
if [[ "${DEBUG_LATBENCH_SBATCH:-0}" == "1" ]]; then
  set -x
fi

cd ~/Craftax_Baselines
source ~/.bashrc >/dev/null 2>&1 || true

USER_NAME="${USER:-${LOGNAME:-geney}}"
SRC_ENV_PATH="${SRC_ENV_PATH:-/data/user_data/${USER_NAME}/.conda/envs/craftax_fast_llm}"
TARGET_VLLM_VERSION="${TARGET_VLLM_VERSION:-0.17.1}"
OVERLAY_SITE_MODE="${OVERLAY_SITE_MODE:-system}"  # system|isolated
if [[ -z "${OVERLAY_ROOT:-}" ]]; then
  OVERLAY_ROOT="/scratch/${USER_NAME}/latbench_overlay_${SLURM_JOB_ID}"
fi
if ! mkdir -p "${OVERLAY_ROOT}" 2>/dev/null; then
  OVERLAY_ROOT="${SLURM_TMPDIR:-/tmp}/latbench_overlay_${SLURM_JOB_ID}"
  mkdir -p "${OVERLAY_ROOT}"
fi
OVERLAY_ENV_PATH="${OVERLAY_ENV_PATH:-${OVERLAY_ROOT}/venv}"
TMPDIR_LOCAL="${OVERLAY_ROOT}/tmp"
XDG_CACHE_HOME="${OVERLAY_ROOT}/xdg_cache"
PIP_CACHE_DIR="${OVERLAY_ROOT}/pip_cache"
UV_CACHE_DIR="${OVERLAY_ROOT}/uv_cache"

if [[ ! -d "${SRC_ENV_PATH}" ]]; then
  echo "ERROR: source env missing: ${SRC_ENV_PATH}"
  exit 2
fi
mkdir -p "${TMPDIR_LOCAL}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}"
export TMPDIR="${TMPDIR_LOCAL}"
export XDG_CACHE_HOME
export PIP_CACHE_DIR
export UV_CACHE_DIR
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
echo "[info] script_started=1"
echo "[info] host=$(hostname)"
echo "[info] src_env=${SRC_ENV_PATH}"
echo "[info] overlay_root=${OVERLAY_ROOT}"
echo "[info] overlay_env=${OVERLAY_ENV_PATH}"
echo "[info] target_vllm_version=${TARGET_VLLM_VERSION}"
echo "[info] overlay_site_mode=${OVERLAY_SITE_MODE}"
echo "[info] tmpdir=${TMPDIR}"
echo "[info] xdg_cache_home=${XDG_CACHE_HOME}"
echo "[info] pip_cache_dir=${PIP_CACHE_DIR}"
echo "[info] pip_no_cache_dir=${PIP_NO_CACHE_DIR}"

REBUILD_OVERLAY=1
if [[ -x "${OVERLAY_ENV_PATH}/bin/python" ]]; then
  if "${OVERLAY_ENV_PATH}/bin/python" - <<PY >/dev/null 2>&1
import sys
try:
    import vllm
    import vllm_hidden_states_extractor
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if getattr(vllm, "__version__", "") == "${TARGET_VLLM_VERSION}" else 1)
PY
  then
    REBUILD_OVERLAY=0
  fi
fi

if [[ "${REBUILD_OVERLAY}" == "1" ]]; then
  echo "[step] rebuild overlay env"
  rm -rf "${OVERLAY_ENV_PATH}"
  if [[ "${OVERLAY_SITE_MODE}" == "system" ]]; then
    echo "[step] create overlay venv with system site-packages"
    "${SRC_ENV_PATH}/bin/python" -m venv --system-site-packages "${OVERLAY_ENV_PATH}"
  else
    echo "[step] create isolated overlay venv"
    "${SRC_ENV_PATH}/bin/python" -m venv "${OVERLAY_ENV_PATH}"
  fi

  echo "[step] upgrade pip tooling in overlay env"
  PIP_DISABLE_PIP_VERSION_CHECK=1 "${OVERLAY_ENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel

  echo "[step] install benchmark deps in overlay env"
  PIP_DISABLE_PIP_VERSION_CHECK=1 "${OVERLAY_ENV_PATH}/bin/python" -m pip install \
    "vllm==${TARGET_VLLM_VERSION}" \
    "accelerate>=0.33.0" \
    "safetensors>=0.4.0"

  echo "[step] install hidden-state extractor in overlay env"
  PIP_DISABLE_PIP_VERSION_CHECK=1 "${OVERLAY_ENV_PATH}/bin/python" -m pip install --no-deps \
    "git+https://github.com/fynnsu/vllm-hidden-states-extractor.git"
else
  echo "[step] reuse existing overlay env"
fi

echo "[step] overlay probe"
"${OVERLAY_ENV_PATH}/bin/python" - <<'PY'
import os
import sys
print("python_exe", sys.executable)
import vllm
print("vllm_version", vllm.__version__)
vdir = os.path.dirname(vllm.__file__)
print("has_qwen3_5_py", os.path.exists(os.path.join(vdir, "model_executor/models/qwen3_5.py")))
import transformers
print("transformers_version", transformers.__version__)
import vllm_hidden_states_extractor
print("hidden_plugin_path", vllm_hidden_states_extractor.__file__)
PY

if [[ -z "${TMP_ROOT:-}" ]]; then
  TMP_ROOT="/scratch/${USER_NAME}/latbench_${SLURM_JOB_ID}"
fi
if ! mkdir -p "${TMP_ROOT}" 2>/dev/null; then
  TMP_ROOT="${SLURM_TMPDIR:-/tmp}/latbench_${SLURM_JOB_ID}"
  mkdir -p "${TMP_ROOT}"
fi

export PATH="${OVERLAY_ENV_PATH}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
HF_HOME="${OVERLAY_ROOT}/hf_home"
mkdir -p "${HF_HOME}"
export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

RUN_TAG="${RUN_TAG:-latbench_plugin_vs_hf}" \
QWEN35_BACKEND="${QWEN35_BACKEND:-vllm_hidden}" \
RUN_QWEN4B="${RUN_QWEN4B:-1}" \
RUN_QWEN8B="${RUN_QWEN8B:-1}" \
RUN_QWEN14B="${RUN_QWEN14B:-1}" \
RUN_QWEN9B="${RUN_QWEN9B:-1}" \
RUN_QWEN27B="${RUN_QWEN27B:-1}" \
RUN_QWEN4B_HF="${RUN_QWEN4B_HF:-1}" \
RUN_QWEN8B_HF="${RUN_QWEN8B_HF:-1}" \
RUN_QWEN14B_HF="${RUN_QWEN14B_HF:-1}" \
RUN_QWEN9B_HF="${RUN_QWEN9B_HF:-1}" \
RUN_QWEN27B_HF="${RUN_QWEN27B_HF:-1}" \
BENCHMARK_MODE="${BENCHMARK_MODE:-hidden_only}" \
WARMUP_PROMPTS_PER_VARIANT="${WARMUP_PROMPTS_PER_VARIANT:-1}" \
REPEATS="${REPEATS:-1}" \
MAX_TOKENS="${MAX_TOKENS:-256}" \
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}" \
PROMPT_STATE_DIR="${PROMPT_STATE_DIR:-analysis/future_imagination_latency_bench/prompts_single/predict_state_only}" \
PROMPT_HISTORY_DIR="${PROMPT_HISTORY_DIR:-analysis/future_imagination_latency_bench/prompts_single/predict_history_k5}" \
TMP_ROOT="${TMP_ROOT}" \
ENABLE_HYBRID_KV_CACHE_MANAGER="${ENABLE_HYBRID_KV_CACHE_MANAGER:-auto}" \
REQUIRE_NONPRIMARY_ENV_FOR_QWEN35="${REQUIRE_NONPRIMARY_ENV_FOR_QWEN35:-1}" \
GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
SAFE_MODE="${SAFE_MODE:-0}" \
env -u BASH_ENV bash --noprofile --norc scripts/shell/run_future_latency_benchmark_models.sh

echo "[done] plugin-vs-hf latency benchmark finished"
