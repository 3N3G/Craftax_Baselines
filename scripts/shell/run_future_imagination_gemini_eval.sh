#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model_id> <model_tag>"
  echo "Example: $0 gemini-3-flash-preview gemini3flashpreview-p7"
  exit 2
fi

MODEL_ID="$1"
MODEL_TAG="$2"

TRAJ_DIR="play_data/trajectory_records/traj_20260311_100410"
CFG_PATH="${RUN_CONFIG_PATH:-configs/future_imagination/run_config_predict_state_and_historyk5_key7_concise_traj_20260311_100410.json}"
BASE_OUT="analysis/future_imagination"
API_KEY_ENV="${API_KEY_ENV:-GEMINI_API_KEY}"

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "ERROR: ${API_KEY_ENV} is not set"
  exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BASE_OUT}/${RUN_TS}_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_${MODEL_TAG}"

echo "[info] model=${MODEL_ID} tag=${MODEL_TAG}"
echo "[info] out_dir=${OUT_DIR}"
echo "[info] config=${CFG_PATH}"
echo "[info] api_key_env=${API_KEY_ENV}"

python3 - <<'PY'
import importlib
mods = ["numpy"]
for m in mods:
    importlib.import_module(m)
print("python_preflight_ok")
PY

python3 scripts/future_imagination_eval.py \
  --trajectory-dir "${TRAJ_DIR}" \
  --config "${CFG_PATH}" \
  --provider gemini \
  --model "${MODEL_ID}" \
  --api-key-env "${API_KEY_ENV}" \
  --output-dir "${OUT_DIR}" \
  --store-full-prompts \
  --resume

echo "[done] out_dir=${OUT_DIR}"
