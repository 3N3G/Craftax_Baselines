#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model_id> <model_tag>"
  echo "Example: $0 Qwen/Qwen3-4B qwen3-4b"
  exit 2
fi

MODEL_ID="$1"
MODEL_TAG="$2"

TRAJ_DIR="play_data/trajectory_records/traj_20260311_100410"
CFG_PATH="${RUN_CONFIG_PATH:-configs/future_imagination/run_config_predict_state_and_historyk5_key7_traj_20260311_100410.json}"
BASE_OUT="analysis/future_imagination"
HF_DTYPE="${HF_DTYPE:-bfloat16}"
HF_DEVICE_MAP="${HF_DEVICE_MAP:-auto}"
INSTALL_TRANSFORMERS_MAIN="${INSTALL_TRANSFORMERS_MAIN:-0}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BASE_OUT}/${RUN_TS}_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_${MODEL_TAG}"

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  TMP_ROOT="${SLURM_TMPDIR}/future_imagination_hf_${MODEL_TAG}_${SLURM_JOB_ID:-nojob}"
else
  TMP_ROOT="/scratch/${USER}/future_imagination_hf_${MODEL_TAG}_${SLURM_JOB_ID:-nojob}"
fi
mkdir -p "${TMP_ROOT}"

export TMPDIR="${TMP_ROOT}/tmp"
export XDG_CACHE_HOME="${TMP_ROOT}/xdg_cache"
export PIP_CACHE_DIR="${TMP_ROOT}/pip_cache"
export HF_HOME="${TMP_ROOT}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_XET=1
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[info] model=${MODEL_ID} tag=${MODEL_TAG}"
echo "[info] out_dir=${OUT_DIR}"
echo "[info] config=${CFG_PATH}"
echo "[info] hf_dtype=${HF_DTYPE} hf_device_map=${HF_DEVICE_MAP}"
echo "[info] install_transformers_main=${INSTALL_TRANSFORMERS_MAIN}"
echo "[info] tmp_root=${TMP_ROOT}"
echo "[info] hf_home=${HF_HOME}"

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
  MODEL_ID_ENV="${MODEL_ID}" python3 - <<'PY'
import importlib
import os
from transformers import AutoConfig
print("transformers_overlay_version", importlib.import_module("transformers").__version__)
cfg = AutoConfig.from_pretrained(os.environ["MODEL_ID_ENV"])
print("qwen35_config_ok", cfg.model_type)
PY
fi

python3 - <<'PY'
import importlib
mods = ["numpy", "torch", "transformers"]
for m in mods:
    importlib.import_module(m)
print("python_preflight_ok")
PY

python3 scripts/future_imagination_eval.py \
  --trajectory-dir "${TRAJ_DIR}" \
  --config "${CFG_PATH}" \
  --provider hf_local \
  --model "${MODEL_ID}" \
  --hf-device-map "${HF_DEVICE_MAP}" \
  --hf-dtype "${HF_DTYPE}" \
  --no-hf-enable-thinking \
  --hf-trim-to-headline \
  --output-dir "${OUT_DIR}" \
  --store-full-prompts \
  --resume

echo "[done] out_dir=${OUT_DIR}"
