#!/usr/bin/env bash
set -euo pipefail

# Clone a known-good vLLM environment before experimenting with Qwen3.5 support.
# This avoids mutating the primary env used by active jobs.

SRC_ENV_PATH="${SRC_ENV_PATH:-/data/user_data/${USER}/.conda/envs/craftax_fast_llm}"
DST_ENV_PATH="${DST_ENV_PATH:-/data/user_data/${USER}/.conda/envs/craftax_fast_llm_qwen35exp}"
UPGRADE_VLLM="${UPGRADE_VLLM:-0}"
TARGET_VLLM_VERSION="${TARGET_VLLM_VERSION:-0.17.1}"
CONDA_BIN="${CONDA_BIN:-/home/${USER}/anaconda3/bin/conda}"

echo "[info] host=$(hostname)"
echo "[info] src_env=${SRC_ENV_PATH}"
echo "[info] dst_env=${DST_ENV_PATH}"
echo "[info] upgrade_vllm=${UPGRADE_VLLM}"
echo "[info] target_vllm_version=${TARGET_VLLM_VERSION}"
echo "[info] conda_bin=${CONDA_BIN}"

if [[ ! -d "${SRC_ENV_PATH}" ]]; then
  echo "[error] missing source env: ${SRC_ENV_PATH}"
  exit 2
fi

if [[ "${SRC_ENV_PATH}" == "${DST_ENV_PATH}" ]]; then
  echo "[error] src_env and dst_env must differ"
  exit 3
fi

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[error] conda binary not found or not executable: ${CONDA_BIN}"
  exit 4
fi

if [[ ! -d "${DST_ENV_PATH}" ]]; then
  echo "[step] cloning env"
  "${CONDA_BIN}" create -y -p "${DST_ENV_PATH}" --clone "${SRC_ENV_PATH}"
else
  echo "[step] clone already exists; skipping clone"
fi

if [[ "${UPGRADE_VLLM}" == "1" ]]; then
  echo "[step] upgrading vllm in clone"
  "${DST_ENV_PATH}/bin/python" -m pip install --upgrade "vllm==${TARGET_VLLM_VERSION}"
fi

echo "[step] probe cloned env"
"${DST_ENV_PATH}/bin/python" - <<'PY'
import os
import sys

print("python_exe", sys.executable)
try:
    import vllm
    vllm_dir = os.path.dirname(vllm.__file__)
    print("vllm_version", vllm.__version__)
    print("qwen3_5_model_file_exists", os.path.exists(os.path.join(vllm_dir, "model_executor/models/qwen3_5.py")))
except Exception as exc:
    print("vllm_import_error", repr(exc))

try:
    import vllm_hidden_states_extractor
    print("hidden_extractor_path", vllm_hidden_states_extractor.__file__)
except Exception as exc:
    print("hidden_extractor_import_error", repr(exc))
PY

echo "[done] clone/probe complete"
