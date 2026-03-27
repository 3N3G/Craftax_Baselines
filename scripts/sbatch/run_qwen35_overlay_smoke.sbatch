#!/bin/bash
#SBATCH --job-name=q35_overlay_smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/q35_overlay_smoke_%j.out
#SBATCH --error=logs/q35_overlay_smoke_%j.err

set -eo pipefail

cd ~/Craftax_Baselines
source ~/.bashrc >/dev/null 2>&1 || true

SRC_ENV_PATH="${SRC_ENV_PATH:-/data/user_data/${USER}/.conda/envs/craftax_fast_llm}"
TARGET_VLLM_VERSION="${TARGET_VLLM_VERSION:-0.17.1}"
OVERLAY_ROOT="${OVERLAY_ROOT:-${SLURM_TMPDIR:-/scratch/${USER}/q35_overlay_${SLURM_JOB_ID}}}"
OVERLAY_ENV_PATH="${OVERLAY_ENV_PATH:-${OVERLAY_ROOT}/venv}"
RUN_TAG="${RUN_TAG:-q35_overlay_smoke}"
PROMPT_ROOT_DEFAULT="analysis/future_imagination/20260316_111259_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3-4b-p6/prompts"
PROMPT_ROOT="${PROMPT_ROOT:-${PROMPT_ROOT_DEFAULT}}"

if [[ ! -d "${SRC_ENV_PATH}" ]]; then
  echo "ERROR: source env missing: ${SRC_ENV_PATH}"
  exit 2
fi

mkdir -p "${OVERLAY_ROOT}"
echo "[info] host=$(hostname)"
echo "[info] src_env=${SRC_ENV_PATH}"
echo "[info] overlay_root=${OVERLAY_ROOT}"
echo "[info] overlay_env=${OVERLAY_ENV_PATH}"
echo "[info] target_vllm_version=${TARGET_VLLM_VERSION}"

echo "[step] create overlay venv with system site-packages"
"${SRC_ENV_PATH}/bin/python" -m venv --system-site-packages "${OVERLAY_ENV_PATH}"

echo "[step] probe overlay before vllm upgrade"
"${OVERLAY_ENV_PATH}/bin/python" - <<'PY'
import os
import sys

print("python_exe", sys.executable)
try:
    import vllm
    vdir = os.path.dirname(vllm.__file__)
    print("vllm_version", vllm.__version__)
    print("has_qwen3_5_py", os.path.exists(os.path.join(vdir, "model_executor/models/qwen3_5.py")))
except Exception as exc:
    print("vllm_import_error", repr(exc))

try:
    import vllm_hidden_states_extractor
    print("hidden_plugin_path", vllm_hidden_states_extractor.__file__)
except Exception as exc:
    print("hidden_plugin_import_error", repr(exc))
PY

echo "[step] upgrade vllm in overlay venv"
"${OVERLAY_ENV_PATH}/bin/python" -m pip install --upgrade "vllm==${TARGET_VLLM_VERSION}"

echo "[step] install hidden-state plugin in overlay venv"
"${OVERLAY_ENV_PATH}/bin/python" -m pip install --no-deps --upgrade \
  "git+https://github.com/fynnsu/vllm-hidden-states-extractor.git"

echo "[step] probe overlay after vllm upgrade"
"${OVERLAY_ENV_PATH}/bin/python" - <<'PY'
import os
import sys

print("python_exe", sys.executable)
try:
    import vllm
    vdir = os.path.dirname(vllm.__file__)
    print("vllm_version", vllm.__version__)
    print("has_qwen3_5_py", os.path.exists(os.path.join(vdir, "model_executor/models/qwen3_5.py")))
except Exception as exc:
    print("vllm_import_error", repr(exc))

try:
    import vllm_hidden_states_extractor
    print("hidden_plugin_path", vllm_hidden_states_extractor.__file__)
except Exception as exc:
    print("hidden_plugin_import_error", repr(exc))
PY

SMOKE_PROMPTS="${OVERLAY_ROOT}/smoke_prompts"
SMOKE_STATE="${SMOKE_PROMPTS}/predict_state_only"
SMOKE_HISTORY="${SMOKE_PROMPTS}/predict_history_k5"
mkdir -p "${SMOKE_STATE}" "${SMOKE_HISTORY}"

cp "${PROMPT_ROOT}/predict_state_only/t_02195.txt" "${SMOKE_STATE}/"
cp "${PROMPT_ROOT}/predict_history_k5/t_02195.txt" "${SMOKE_HISTORY}/"

export PATH="${OVERLAY_ENV_PATH}/bin:${PATH}"
ENABLE_QWEN35_RUNTIME_PATCHES="${ENABLE_QWEN35_RUNTIME_PATCHES:-1}"

echo "[step] run qwen3.5 9b smoke benchmark (single prompt per variant)"
RUN_TAG="${RUN_TAG}" \
RUN_QWEN4B=0 \
RUN_QWEN14B=0 \
RUN_QWEN9B=1 \
RUN_QWEN27B=0 \
REPEATS=1 \
MAX_TOKENS=64 \
REQUEST_TIMEOUT_S=240 \
PROMPT_STATE_DIR="${SMOKE_STATE}" \
PROMPT_HISTORY_DIR="${SMOKE_HISTORY}" \
ENABLE_QWEN35_RUNTIME_PATCHES="${ENABLE_QWEN35_RUNTIME_PATCHES}" \
REQUIRE_NONPRIMARY_ENV_FOR_QWEN35=1 \
ALLOW_PRIMARY_PLUGIN_PATCH=0 \
GEMINI_API_KEY="" \
bash scripts/shell/run_future_latency_benchmark_models.sh

echo "[done] qwen3.5 overlay smoke finished"
