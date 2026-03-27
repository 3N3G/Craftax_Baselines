#!/usr/bin/env bash
set -euo pipefail
if [[ "${DEBUG_LATBENCH_SH:-0}" == "1" ]]; then
  set -x
fi

# Runs latency benchmarks for:
#   - Gemini Flash API
#   - Qwen3-4B / 8B / 14B (vLLM hidden plugin and optional HF local)
#   - Qwen3.5-9B / 27B (optional HF local and/or vLLM hidden plugin)
#
# Intended to run on a Babel compute node with GPUs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_TAG="${RUN_TAG:-latbench}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="analysis/future_imagination_latency_bench/${TS}_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

PROMPT_ROOT="${PROMPT_ROOT:-analysis/future_imagination/20260316_111259_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3-4b-p6/prompts}"
PROMPT_STATE_DIR="${PROMPT_STATE_DIR:-${PROMPT_ROOT}/predict_state_only}"
PROMPT_HISTORY_DIR="${PROMPT_HISTORY_DIR:-${PROMPT_ROOT}/predict_history_k5}"

MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.95}"
REPEATS="${REPEATS:-1}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BENCHMARK_MODE="${BENCHMARK_MODE:-hidden_only}"
WARMUP_PROMPTS_PER_VARIANT="${WARMUP_PROMPTS_PER_VARIANT:-1}"
SAFE_MODE="${SAFE_MODE:-0}"
INSTALL_TRANSFORMERS_MAIN="${INSTALL_TRANSFORMERS_MAIN:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ENABLE_QWEN35_ALIAS_PATCH="${ENABLE_QWEN35_ALIAS_PATCH:-1}"
ENABLE_QWEN35_RUNTIME_PATCHES="${ENABLE_QWEN35_RUNTIME_PATCHES:-auto}"
FORCE_QWEN35_COMPAT_MODEL="${FORCE_QWEN35_COMPAT_MODEL:-0}"
PRIMARY_ENV_PATH="${PRIMARY_ENV_PATH:-/data/user_data/${USER}/.conda/envs/craftax_fast_llm}"
REQUIRE_NONPRIMARY_ENV_FOR_QWEN35="${REQUIRE_NONPRIMARY_ENV_FOR_QWEN35:-1}"
ALLOW_PRIMARY_PLUGIN_PATCH="${ALLOW_PRIMARY_PLUGIN_PATCH:-0}"
QWEN35_BACKEND="${QWEN35_BACKEND:-hf_local}"  # hf_local|vllm_hidden|vllm_offline
QWEN35_HF_DTYPE="${QWEN35_HF_DTYPE:-bfloat16}"
QWEN35_HF_DEVICE_MAP="${QWEN35_HF_DEVICE_MAP:-auto}"
QWEN35_HF_ATTN_IMPL="${QWEN35_HF_ATTN_IMPL:-}"
QWEN35_HF_TRUST_REMOTE_CODE="${QWEN35_HF_TRUST_REMOTE_CODE:-1}"
QWEN35_HF_CACHE_ROOT="${QWEN35_HF_CACHE_ROOT:-}"
QWEN35_VLLM_OFFLINE_DTYPE="${QWEN35_VLLM_OFFLINE_DTYPE:-bfloat16}"
QWEN35_VLLM_OFFLINE_TRUST_REMOTE_CODE="${QWEN35_VLLM_OFFLINE_TRUST_REMOTE_CODE:-1}"
QWEN35_VLLM_OFFLINE_ENFORCE_EAGER="${QWEN35_VLLM_OFFLINE_ENFORCE_EAGER:-0}"
QWEN35_VLLM_OFFLINE_SPEC_NUM_TOKENS="${QWEN35_VLLM_OFFLINE_SPEC_NUM_TOKENS:-1}"
QWEN35_VLLM_OFFLINE_PREFIX_CACHING="${QWEN35_VLLM_OFFLINE_PREFIX_CACHING:-0}"
QWEN35_VLLM_OFFLINE_CHUNKED_PREFILL="${QWEN35_VLLM_OFFLINE_CHUNKED_PREFILL:-0}"
# Default HMA off for qwen3.5 offline hidden extraction. With the current
# connector/docs path on vLLM>=0.17, HMA can get past startup but still die in
# save_kv_layer with malformed slot mappings under hybrid cache groups.
QWEN35_VLLM_OFFLINE_ENABLE_HMA="${QWEN35_VLLM_OFFLINE_ENABLE_HMA:-0}"
QWEN35_VLLM_OFFLINE_CONNECTOR_MODULE_PATH="${QWEN35_VLLM_OFFLINE_CONNECTOR_MODULE_PATH:-utils.vllm_hidden_connector}"
QWEN35_VLLM_OFFLINE_CONNECTOR_MODE="${QWEN35_VLLM_OFFLINE_CONNECTOR_MODE:-last_token}"
QWEN35_MIRROR_PROVIDER="${QWEN35_MIRROR_PROVIDER:-auto}"  # auto|modelscope|none
QWEN35_LOCAL_MIRROR_ROOT="${QWEN35_LOCAL_MIRROR_ROOT:-}"
QWEN35_MODELSCOPE_MAX_WORKERS="${QWEN35_MODELSCOPE_MAX_WORKERS:-4}"

QWEN4B_MODEL_ID="${QWEN4B_MODEL_ID:-Qwen/Qwen3-4B}"
QWEN4B_CONFIG="${QWEN4B_CONFIG:-configs/vllm_hidden_qwen4b}"
QWEN8B_MODEL_ID="${QWEN8B_MODEL_ID:-Qwen/Qwen3-8B}"
QWEN8B_CONFIG_OUT="${QWEN8B_CONFIG_OUT:-configs/vllm_hidden_qwen3_8b_auto4}"
QWEN14B_MODEL_ID="${QWEN14B_MODEL_ID:-Qwen/Qwen3-14B}"
QWEN9B_MODEL_ID="${QWEN9B_MODEL_ID:-Qwen/Qwen3.5-9B}"
QWEN27B_MODEL_ID="${QWEN27B_MODEL_ID:-Qwen/Qwen3.5-27B}"
QWEN14B_CONFIG_OUT="${QWEN14B_CONFIG_OUT:-configs/vllm_hidden_qwen3_14b_auto4}"
QWEN9B_CONFIG_OUT="${QWEN9B_CONFIG_OUT:-configs/vllm_hidden_qwen35_9b_auto4}"
QWEN27B_CONFIG_OUT="${QWEN27B_CONFIG_OUT:-configs/vllm_hidden_qwen35_27b_auto4}"
QWEN9B_COMPAT_MODEL_DIR="${QWEN9B_COMPAT_MODEL_DIR:-}"
QWEN27B_COMPAT_MODEL_DIR="${QWEN27B_COMPAT_MODEL_DIR:-}"
QWEN9B_LOCAL_MODEL_DIR="${QWEN9B_LOCAL_MODEL_DIR:-}"
QWEN27B_LOCAL_MODEL_DIR="${QWEN27B_LOCAL_MODEL_DIR:-}"

QWEN4B_PORT="${QWEN4B_PORT:-8101}"
QWEN8B_PORT="${QWEN8B_PORT:-8105}"
QWEN14B_PORT="${QWEN14B_PORT:-8104}"
QWEN9B_PORT="${QWEN9B_PORT:-8102}"
QWEN27B_PORT="${QWEN27B_PORT:-8103}"

QWEN4B_GPU_UTIL="${QWEN4B_GPU_UTIL:-0.70}"
QWEN8B_GPU_UTIL="${QWEN8B_GPU_UTIL:-0.80}"
QWEN14B_GPU_UTIL="${QWEN14B_GPU_UTIL:-0.90}"
QWEN9B_GPU_UTIL="${QWEN9B_GPU_UTIL:-0.86}"
QWEN27B_GPU_UTIL="${QWEN27B_GPU_UTIL:-0.92}"
RUN_QWEN4B="${RUN_QWEN4B:-1}"
RUN_QWEN8B="${RUN_QWEN8B:-1}"
RUN_QWEN14B="${RUN_QWEN14B:-1}"
RUN_QWEN9B="${RUN_QWEN9B:-1}"
RUN_QWEN27B="${RUN_QWEN27B:-1}"
RUN_QWEN4B_HF="${RUN_QWEN4B_HF:-0}"
RUN_QWEN8B_HF="${RUN_QWEN8B_HF:-0}"
RUN_QWEN14B_HF="${RUN_QWEN14B_HF:-0}"
RUN_QWEN9B_HF="${RUN_QWEN9B_HF:-0}"
RUN_QWEN27B_HF="${RUN_QWEN27B_HF:-0}"
QWEN8B_TP="${QWEN8B_TP:-1}"
QWEN14B_TP="${QWEN14B_TP:-1}"
QWEN9B_TP="${QWEN9B_TP:-1}"
QWEN27B_TP="${QWEN27B_TP:-2}"
# The hidden-state connector path is still brittle under HMA on vLLM>=0.17.
# Default to the stable non-HMA path unless explicitly overridden.
ENABLE_HYBRID_KV_CACHE_MANAGER="${ENABLE_HYBRID_KV_CACHE_MANAGER:-auto}"

GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-flash-preview}"
GEMINI_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"

if [[ ! -d "${PROMPT_STATE_DIR}" || ! -d "${PROMPT_HISTORY_DIR}" ]]; then
  echo "ERROR: prompt dirs not found:"
  echo "  state=${PROMPT_STATE_DIR}"
  echo "  history=${PROMPT_HISTORY_DIR}"
  exit 1
fi

if [[ -n "${TMP_ROOT:-}" ]]; then
  TMP_ROOT="${TMP_ROOT}"
elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
  TMP_ROOT="${SLURM_TMPDIR}/latbench_${TS}"
else
  TMP_ROOT="/scratch/${USER}/latbench_${TS}"
fi
mkdir -p "${TMP_ROOT}"

MODELSCOPE_PYTHONPATH=""

choose_qwen35_local_mirror_root() {
  local required_bytes="${1:-0}"
  local candidates=()
  if [[ -n "${QWEN35_LOCAL_MIRROR_ROOT}" ]]; then
    candidates+=("${QWEN35_LOCAL_MIRROR_ROOT}")
  else
    candidates+=(
      "/data/group_data/rl/${USER}/model_mirrors"
      "/data/user_data/${USER}/model_mirrors"
      "${TMP_ROOT}/model_mirrors"
    )
  fi

  local cand=""
  local avail=""
  local probe=""
  for cand in "${candidates[@]}"; do
    mkdir -p "${cand}" 2>/dev/null || continue
    probe="${cand}/.codex_probe_$$"
    if ! echo test > "${probe}" 2>/dev/null; then
      continue
    fi
    rm -f "${probe}"

    avail="$(df -PB1 "${cand}" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ -n "${avail}" && "${avail}" =~ ^[0-9]+$ && "${required_bytes}" =~ ^[0-9]+$ ]]; then
      if (( avail < required_bytes )); then
        echo "[warn] skip mirror root ${cand}: available=${avail} < required=${required_bytes}" >&2
        continue
      fi
    fi
    echo "${cand}"
    return 0
  done
  return 1
}

ensure_modelscope_overlay() {
  if [[ -n "${MODELSCOPE_PYTHONPATH}" && -f "${MODELSCOPE_PYTHONPATH}/modelscope/__init__.py" ]]; then
    return 0
  fi
  MODELSCOPE_PYTHONPATH="${TMP_ROOT}/py_modelscope"
  if [[ ! -f "${MODELSCOPE_PYTHONPATH}/modelscope/__init__.py" ]]; then
    echo "[step] install modelscope downloader overlay"
    python3 -m pip install --quiet --target "${MODELSCOPE_PYTHONPATH}" modelscope
  fi
}

verify_qwen35_local_model_dir() {
  local model_dir="$1"
  python3 - "${model_dir}" <<'PY' >/dev/null
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cfg = root / "config.json"
idx = root / "model.safetensors.index.json"
if not cfg.exists() or not idx.exists():
    raise SystemExit(1)

data = json.loads(idx.read_text(encoding="utf-8"))
weight_map = data.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit(1)

for rel_name in sorted(set(weight_map.values())):
    shard = root / rel_name
    if not shard.exists() or shard.stat().st_size < 1024 * 1024:
        raise SystemExit(1)
    with shard.open("rb") as fh:
        head = fh.read(64)
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise SystemExit(1)
PY
}

mirror_qwen35_model_if_needed() {
  local model_id="$1"
  local local_dir="$2"
  local label="$3"

  if [[ "${QWEN35_MIRROR_PROVIDER}" == "none" ]]; then
    return 0
  fi
  if [[ "${model_id}" != Qwen/Qwen3.5-* ]]; then
    return 0
  fi
  if [[ -z "${local_dir}" ]]; then
    echo "[error] empty local_dir for ${label}" >&2
    return 1
  fi
  if verify_qwen35_local_model_dir "${local_dir}"; then
    echo "[info] reuse local ${label} mirror: ${local_dir}"
    return 0
  fi

  local tmp_dir="${local_dir}.downloading"
  if verify_qwen35_local_model_dir "${tmp_dir}"; then
    rm -rf "${local_dir}"
    mv "${tmp_dir}" "${local_dir}"
    echo "[info] finalized completed ${label} mirror from ${tmp_dir}"
    return 0
  fi
  mkdir -p "${tmp_dir}"

  ensure_modelscope_overlay
  echo "[step] download ${label} from ModelScope -> ${local_dir}"
  PYTHONPATH="${MODELSCOPE_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" \
  MODEL_ID="${model_id}" \
  LOCAL_DIR="${tmp_dir}" \
  MODELSCOPE_MAX_WORKERS="${QWEN35_MODELSCOPE_MAX_WORKERS}" \
  python3 - <<'PY'
import os
from modelscope import snapshot_download

snapshot_download(
    model_id=os.environ["MODEL_ID"],
    local_dir=os.environ["LOCAL_DIR"],
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.txt",
        "*.tiktoken",
        "*.model",
        "*.py",
    ],
    max_workers=int(os.environ.get("MODELSCOPE_MAX_WORKERS", "4")),
)
PY

  rm -rf "${local_dir}"
  mv "${tmp_dir}" "${local_dir}"
  verify_qwen35_local_model_dir "${local_dir}"
  echo "[info] completed local ${label} mirror: ${local_dir}"
}

# Force all HF/transformers caches inside the job temp root to avoid inherited
# quota-limited cache paths from shell startup files.
export HF_HOME="${TMP_ROOT}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
# Xet-backed downloads have intermittently failed on Babel nodes for large
# checkpoints; force plain hub download path for stability.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ -z "${QWEN9B_COMPAT_MODEL_DIR}" ]]; then
  QWEN9B_COMPAT_MODEL_DIR="${TMP_ROOT}/qwen35_9b_compat"
fi
if [[ -z "${QWEN27B_COMPAT_MODEL_DIR}" ]]; then
  QWEN27B_COMPAT_MODEL_DIR="${TMP_ROOT}/qwen35_27b_compat"
fi
if [[ -z "${QWEN35_HF_CACHE_ROOT}" ]]; then
  QWEN35_HF_CACHE_ROOT="${TMP_ROOT}/hf_cache"
fi

USE_QWEN35_VLLM_HIDDEN=0
USE_QWEN35_VLLM_OFFLINE=0
if [[ "${QWEN35_BACKEND}" == "vllm_hidden" ]]; then
  USE_QWEN35_VLLM_HIDDEN=1
elif [[ "${QWEN35_BACKEND}" == "vllm_offline" ]]; then
  USE_QWEN35_VLLM_OFFLINE=1
elif [[ "${QWEN35_BACKEND}" != "hf_local" ]]; then
  echo "[error] unsupported QWEN35_BACKEND=${QWEN35_BACKEND}; expected hf_local, vllm_hidden, or vllm_offline"
  exit 1
fi

if [[ "${QWEN35_MIRROR_PROVIDER}" == "auto" ]]; then
  QWEN35_MIRROR_PROVIDER="modelscope"
fi

if [[ "${QWEN35_MIRROR_PROVIDER}" != "none" ]]; then
  if [[ "${RUN_QWEN9B}" == "1" || "${RUN_QWEN9B_HF}" == "1" ]]; then
    if [[ -z "${QWEN9B_LOCAL_MODEL_DIR}" ]]; then
      QWEN35_9B_ROOT="$(choose_qwen35_local_mirror_root 25000000000 || true)"
      if [[ -n "${QWEN35_9B_ROOT}" ]]; then
        QWEN9B_LOCAL_MODEL_DIR="${QWEN35_9B_ROOT}/Qwen3.5-9B"
      fi
    fi
    if [[ -n "${QWEN9B_LOCAL_MODEL_DIR}" ]]; then
      mirror_qwen35_model_if_needed "${QWEN9B_MODEL_ID}" "${QWEN9B_LOCAL_MODEL_DIR}" "Qwen3.5-9B"
      QWEN9B_MODEL_ID="${QWEN9B_LOCAL_MODEL_DIR}"
    fi
  fi

  if [[ "${RUN_QWEN27B}" == "1" || "${RUN_QWEN27B_HF}" == "1" ]]; then
    if [[ -z "${QWEN27B_LOCAL_MODEL_DIR}" ]]; then
      QWEN35_27B_ROOT="$(choose_qwen35_local_mirror_root 70000000000 || true)"
      if [[ -n "${QWEN35_27B_ROOT}" ]]; then
        QWEN27B_LOCAL_MODEL_DIR="${QWEN35_27B_ROOT}/Qwen3.5-27B"
      fi
    fi
    if [[ -n "${QWEN27B_LOCAL_MODEL_DIR}" ]]; then
      mirror_qwen35_model_if_needed "${QWEN27B_MODEL_ID}" "${QWEN27B_LOCAL_MODEL_DIR}" "Qwen3.5-27B"
      QWEN27B_MODEL_ID="${QWEN27B_LOCAL_MODEL_DIR}"
    fi
  fi
fi

USE_ANY_VLLM_RUN=0
if [[ "${RUN_QWEN4B}" == "1" || "${RUN_QWEN8B}" == "1" || "${RUN_QWEN14B}" == "1" || ( ( "${USE_QWEN35_VLLM_HIDDEN}" == "1" || "${USE_QWEN35_VLLM_OFFLINE}" == "1" ) && ( "${RUN_QWEN9B}" == "1" || "${RUN_QWEN27B}" == "1" ) ) ]]; then
  USE_ANY_VLLM_RUN=1
fi

USE_ANY_PLUGIN_RUN=0
if [[ "${RUN_QWEN4B}" == "1" || "${RUN_QWEN8B}" == "1" || "${RUN_QWEN14B}" == "1" || ( "${USE_QWEN35_VLLM_HIDDEN}" == "1" && ( "${RUN_QWEN9B}" == "1" || "${RUN_QWEN27B}" == "1" ) ) ]]; then
  USE_ANY_PLUGIN_RUN=1
fi

DO_QWEN35_PATCH=0
if [[ "${ENABLE_QWEN35_RUNTIME_PATCHES}" == "1" ]]; then
  DO_QWEN35_PATCH=1
elif [[ "${ENABLE_QWEN35_RUNTIME_PATCHES}" == "auto" ]]; then
  if [[ ( "${USE_QWEN35_VLLM_HIDDEN}" == "1" || "${USE_QWEN35_VLLM_OFFLINE}" == "1" ) && ( "${RUN_QWEN9B}" == "1" || "${RUN_QWEN27B}" == "1" ) ]]; then
    DO_QWEN35_PATCH=1
  fi
fi

HF_OVERLAY_PYTHONPATH=""
if [[ "${INSTALL_TRANSFORMERS_MAIN}" == "1" ]]; then
  echo "[step] install source transformers overlay for Qwen3.5 support"
  PY_OVERLAY="${TMP_ROOT}/py_overlay"
  mkdir -p "${PY_OVERLAY}"
  python3 -m pip install --quiet --target "${PY_OVERLAY}" --upgrade \
    "huggingface_hub>=0.34.0" \
    "tokenizers>=0.22.0"
  python3 -m pip install --quiet --target "${PY_OVERLAY}" --upgrade \
    "git+https://github.com/huggingface/transformers.git@main"
  HF_OVERLAY_PYTHONPATH="${PY_OVERLAY}"
  PYTHONPATH="${HF_OVERLAY_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
import importlib
print("transformers_overlay_version", importlib.import_module("transformers").__version__)
print("hf_home", __import__("os").environ.get("HF_HOME", ""))
print("hf_hub_cache", __import__("os").environ.get("HUGGINGFACE_HUB_CACHE", ""))
PY
fi

echo "[info] out_dir=${OUT_DIR}"
echo "[info] prompt_state_dir=${PROMPT_STATE_DIR}"
echo "[info] prompt_history_dir=${PROMPT_HISTORY_DIR}"
echo "[info] tmp_root=${TMP_ROOT}"
echo "[info] qwen9b_compat_model_dir=${QWEN9B_COMPAT_MODEL_DIR}"
echo "[info] qwen27b_compat_model_dir=${QWEN27B_COMPAT_MODEL_DIR}"
echo "[info] qwen35_backend=${QWEN35_BACKEND}"
VLLM_RUNTIME_VERSION=""
if [[ "${USE_ANY_VLLM_RUN}" == "1" ]]; then
  VLLM_RUNTIME_VERSION="$(python3 -c 'import vllm; print(vllm.__version__)')"
  echo "[info] vllm_runtime_version=${VLLM_RUNTIME_VERSION}"
else
  echo "[info] vllm_runtime_version=skipped"
fi

USE_QWEN35_COMPAT_MODEL=0
if [[ "${USE_QWEN35_VLLM_HIDDEN}" == "1" && "${DO_QWEN35_PATCH}" == "1" && "${VLLM_RUNTIME_VERSION}" == 0.14.* ]]; then
  USE_QWEN35_COMPAT_MODEL=1
fi
if [[ "${USE_QWEN35_VLLM_OFFLINE}" == "1" && "${DO_QWEN35_PATCH}" == "1" ]]; then
  USE_QWEN35_COMPAT_MODEL=1
fi
if [[ ( "${USE_QWEN35_VLLM_HIDDEN}" == "1" || "${USE_QWEN35_VLLM_OFFLINE}" == "1" ) && "${FORCE_QWEN35_COMPAT_MODEL}" == "1" ]]; then
  USE_QWEN35_COMPAT_MODEL=1
fi

TRANSFORMERS_ALIAS_PATCH_DIR=""
DO_TRANSFORMERS_ALIAS_PATCH=0
if [[ "${USE_QWEN35_VLLM_HIDDEN}" == "1" && "${ENABLE_QWEN35_ALIAS_PATCH}" == "1" ]]; then
  if [[ "${VLLM_RUNTIME_VERSION}" == 0.14.* ]]; then
    DO_TRANSFORMERS_ALIAS_PATCH=1
  else
    echo "[info] skipping qwen3_5 transformers alias patch for vLLM ${VLLM_RUNTIME_VERSION}"
  fi
fi

if [[ "${DO_TRANSFORMERS_ALIAS_PATCH}" == "1" ]]; then
  TRANSFORMERS_ALIAS_PATCH_DIR="${TMP_ROOT}/py_patch"
  mkdir -p "${TRANSFORMERS_ALIAS_PATCH_DIR}"
  cat > "${TRANSFORMERS_ALIAS_PATCH_DIR}/sitecustomize.py" <<'PY'
def _patch_qwen35_alias():
    try:
        from transformers import AutoConfig
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

        try:
            AutoConfig.register("qwen3_5", Qwen3Config, exist_ok=True)
        except TypeError:
            AutoConfig.register("qwen3_5", Qwen3Config)
    except Exception:
        return


_patch_qwen35_alias()
PY
  echo "[info] transformers_alias_patch_dir=${TRANSFORMERS_ALIAS_PATCH_DIR}"
fi

VLLM_PYTHONPATH_PREFIX=""
if [[ -n "${TRANSFORMERS_ALIAS_PATCH_DIR}" ]]; then
  VLLM_PYTHONPATH_PREFIX="${TRANSFORMERS_ALIAS_PATCH_DIR}"
fi

run_python() {
  if [[ -n "${VLLM_PYTHONPATH_PREFIX}" ]]; then
    PYTHONPATH="${VLLM_PYTHONPATH_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}" python3 "$@"
  else
    python3 "$@"
  fi
}

run_hf_python() {
  if [[ -n "${HF_OVERLAY_PYTHONPATH}" ]]; then
    PYTHONPATH="${HF_OVERLAY_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" python3 "$@"
  else
    python3 "$@"
  fi
}

CURRENT_ENV_PREFIX="$(python3 -c 'import sys; print(sys.prefix)')"
if [[ ( "${USE_QWEN35_VLLM_HIDDEN}" == "1" || "${USE_QWEN35_VLLM_OFFLINE}" == "1" ) && ( "${RUN_QWEN9B}" == "1" || "${RUN_QWEN27B}" == "1" ) ]]; then
  echo "[info] current_env_prefix=${CURRENT_ENV_PREFIX}"
  if [[ "${REQUIRE_NONPRIMARY_ENV_FOR_QWEN35}" == "1" && "${CURRENT_ENV_PREFIX}" == "${PRIMARY_ENV_PATH}" ]]; then
    echo "[error] refusing Qwen3.5 run in primary env: ${PRIMARY_ENV_PATH}"
    echo "[error] clone env first and activate it, or set REQUIRE_NONPRIMARY_ENV_FOR_QWEN35=0 to override."
    exit 1
  fi
fi

if [[ "${USE_QWEN35_VLLM_OFFLINE}" == "1" && ( "${RUN_QWEN9B}" == "1" || "${RUN_QWEN27B}" == "1" ) ]]; then
  echo "[step] patch vLLM get_hf_text_config for dict text_config compatibility (offline qwen3.5)"
  python3 - <<'PY'
import inspect
import pathlib

import vllm.transformers_utils.config as cfg_mod

cfg_path = pathlib.Path(inspect.getfile(cfg_mod))
text = cfg_path.read_text(encoding="utf-8")
if "dict text_config compatibility for speculative config" not in text:
    needle = (
        "    if text_config is not config and not hasattr(text_config, \"num_attention_heads\"):\n"
        "        raise ValueError(\n"
        "            \"The text_config extracted from the model config does not have \"\n"
        "            \"`num_attention_heads` attribute. This indicates a mismatch \"\n"
        "            \"between the model config and vLLM's expectations. Please \"\n"
        "            \"ensure that the model config is compatible with vLLM.\"\n"
        "        )\n"
    )
    replace = (
        "    if text_config is not config and not hasattr(text_config, \"num_attention_heads\"):\n"
        "        # dict text_config compatibility for speculative config\n"
        "        if isinstance(text_config, dict) and text_config.get(\"num_attention_heads\") is not None:\n"
        "            from types import SimpleNamespace\n"
        "            text_config = SimpleNamespace(**text_config)\n"
        "        else:\n"
        "            raise ValueError(\n"
        "                \"The text_config extracted from the model config does not have \"\n"
        "                \"`num_attention_heads` attribute. This indicates a mismatch \"\n"
        "                \"between the model config and vLLM's expectations. Please \"\n"
        "                \"ensure that the model config is compatible with vLLM.\"\n"
        "            )\n"
    )
    if needle not in text:
        raise RuntimeError(f"Could not find get_hf_text_config target in {cfg_path}")
    text = text.replace(needle, replace, 1)
    cfg_path.write_text(text, encoding="utf-8")
print(f"patched_hf_text_config_offline={cfg_path}")
PY

  echo "[step] patch vLLM gpu_model_runner tuple-unpack compatibility (offline qwen3.5)"
  python3 - <<'PY'
import pathlib
import sys

gmr_path = (
    pathlib.Path(sys.prefix)
    / "lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py"
)
if not gmr_path.exists():
    raise RuntimeError(f"Missing gpu_model_runner.py at expected path: {gmr_path}")

text = gmr_path.read_text(encoding="utf-8")

needle_main = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                # True when EAGLE 3 is used.\n"
    "                hidden_states, aux_hidden_states = model_output\n"
    "            else:\n"
)
replace_main = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                # True when EAGLE 3 is used.\n"
    "                if isinstance(model_output, tuple):\n"
    "                    hidden_states = model_output[0]\n"
    "                    aux_hidden_states = model_output[1] if len(model_output) > 1 else None\n"
    "                else:\n"
    "                    hidden_states = model_output\n"
    "                    aux_hidden_states = None\n"
    "            else:\n"
)
if "aux_hidden_states = model_output[1] if len(model_output) > 1 else None" not in text:
    if needle_main not in text:
        raise RuntimeError(f"Could not find main tuple-unpack target in {gmr_path}")
    text = text.replace(needle_main, replace_main, 1)

needle_dummy = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                hidden_states, _ = outputs\n"
    "            else:\n"
)
replace_dummy = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                if isinstance(outputs, tuple):\n"
    "                    hidden_states = outputs[0]\n"
    "                else:\n"
    "                    hidden_states = outputs\n"
    "            else:\n"
)
if "if isinstance(outputs, tuple):\n                    hidden_states = outputs[0]" not in text:
    if needle_dummy not in text:
        raise RuntimeError(f"Could not find dummy tuple-unpack target in {gmr_path}")
    text = text.replace(needle_dummy, replace_dummy, 1)

if "target_hidden_states = [base_hidden for _ in range(num_aux_layers)]" not in text:
    lines = text.splitlines(keepends=True)
    marker_idx = next(
        (i for i, line in enumerate(lines)
         if "aux_hidden_states are required when using `extract_hidden_states`" in line),
        None,
    )
    if marker_idx is None:
        raise RuntimeError(f"Could not find extract_hidden_states error marker in {gmr_path}")

    start_idx = marker_idx
    while start_idx >= 0 and "if not self.use_aux_hidden_state_outputs" not in lines[start_idx]:
        start_idx -= 1
    if start_idx < 0:
        raise RuntimeError(f"Could not find extract_hidden_states guard start in {gmr_path}")

    end_idx = marker_idx
    while end_idx < len(lines) and "target_hidden_states =" not in lines[end_idx]:
        end_idx += 1
    if end_idx >= len(lines):
        raise RuntimeError(f"Could not find extract_hidden_states target_hidden_states line in {gmr_path}")

    indent = lines[start_idx][: len(lines[start_idx]) - len(lines[start_idx].lstrip())]
    replacement_lines = [
        f"{indent}if not self.use_aux_hidden_state_outputs:\n",
        f"{indent}    raise ValueError(\n",
        f"{indent}        \"aux_hidden_states are required when using `extract_hidden_states`\"\n",
        f"{indent}    )\n",
        f"{indent}if aux_hidden_states is None:\n",
        f"{indent}    layer_ids = getattr(\n",
        f"{indent}        self.speculative_config.draft_model_config.hf_config,\n",
        f"{indent}        \"eagle_aux_hidden_state_layer_ids\",\n",
        f"{indent}        None,\n",
        f"{indent}    )\n",
        f"{indent}    num_aux_layers = len(layer_ids) if layer_ids else 1\n",
        f"{indent}    base_hidden = hidden_states[:num_scheduled_tokens]\n",
        f"{indent}    target_hidden_states = [base_hidden for _ in range(num_aux_layers)]\n",
        f"{indent}else:\n",
        f"{indent}    target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]\n",
    ]
    lines[start_idx:end_idx + 1] = replacement_lines
    text = "".join(lines)

gmr_path.write_text(text, encoding="utf-8")
print(f"patched_gpu_model_runner_tuple_unpack_offline={gmr_path}")
PY

  echo "[step] patch vLLM kv page-size unification fallback (offline qwen3.5)"
  python3 - <<'PY'
import pathlib
import sys

kvu_path = (
    pathlib.Path(sys.prefix)
    / "lib/python3.10/site-packages/vllm/v1/core/kv_cache_utils.py"
)
if not kvu_path.exists():
    raise RuntimeError(f"Missing kv_cache_utils.py at expected path: {kvu_path}")

text = kvu_path.read_text(encoding="utf-8")
if "fallback: unify via page_size_padded" not in text:
    needle = (
        "        else:\n"
        "            layer_page_size = layer_spec.page_size_bytes\n"
        "            if max_page_size % layer_page_size != 0:\n"
        "                raise NotImplementedError(\n"
        "                    \"The page size of the layer is not divisible by the \"\n"
        "                    \"maximum page size. Cannot unify by adjusting block_size.\"\n"
        "                )\n"
        "            ratio = max_page_size // layer_page_size\n"
        "            new_block_size = layer_spec.block_size * ratio\n"
        "            new_spec = replace(layer_spec, block_size=new_block_size)\n"
        "            assert new_spec.page_size_bytes == max_page_size\n"
        "            new_kv_cache_spec[layer_name] = new_spec\n"
    )
    replace = (
        "        else:\n"
        "            layer_page_size = layer_spec.page_size_bytes\n"
        "            new_spec = None\n"
        "            # First try the original block-size scaling path.\n"
        "            if max_page_size % layer_page_size == 0:\n"
        "                ratio = max_page_size // layer_page_size\n"
        "                new_block_size = layer_spec.block_size * ratio\n"
        "                cand = replace(layer_spec, block_size=new_block_size)\n"
        "                if cand.page_size_bytes == max_page_size:\n"
        "                    new_spec = cand\n"
        "\n"
        "            if new_spec is None:\n"
        "                # fallback: unify via page_size_padded for specs where\n"
        "                # page_size is not linear in block_size (e.g. mamba/mixed).\n"
        "                if hasattr(layer_spec, \"page_size_padded\"):\n"
        "                    cand = replace(layer_spec, page_size_padded=max_page_size)\n"
        "                    if cand.page_size_bytes == max_page_size:\n"
        "                        new_spec = cand\n"
        "\n"
        "            if new_spec is None:\n"
        "                raise NotImplementedError(\n"
        "                    \"The page size of the layer is not divisible by the \"\n"
        "                    \"maximum page size and page_size_padded fallback failed.\"\n"
        "                )\n"
        "\n"
        "            new_kv_cache_spec[layer_name] = new_spec\n"
    )
    if needle not in text:
        raise RuntimeError(f"Could not find kv page-size unify target in {kvu_path}")
    text = text.replace(needle, replace, 1)
    kvu_path.write_text(text, encoding="utf-8")

print(f"patched_kv_cache_unify_fallback_offline={kvu_path}")
PY

  echo "[step] patch vLLM disabled-HMA hybrid KV fallback (offline qwen3.5)"
  python3 - <<'PY'
import pathlib
import sys

kvu_path = (
    pathlib.Path(sys.prefix)
    / "lib/python3.10/site-packages/vllm/v1/core/kv_cache_utils.py"
)
if not kvu_path.exists():
    raise RuntimeError(f"Missing kv_cache_utils.py at expected path: {kvu_path}")

text = kvu_path.read_text(encoding="utf-8")
new_marker = "fallback: disabled-HMA mixed specs continue to downstream page-size grouping"
old_marker = "fallback: disabled-HMA hybrid specs via page-size unify"
needle = (
    "    if not (\n"
    "        is_kv_cache_spec_uniform(kv_cache_spec)\n"
    "        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)\n"
    "    ):\n"
    "        raise ValueError(\n"
    "            \"Hybrid KV cache manager is disabled but failed to \"\n"
    "            \"convert the KV cache specs to one unified type.\"\n"
    "        )\n"
)
old_replace = (
    "    if not (\n"
    "        is_kv_cache_spec_uniform(kv_cache_spec)\n"
    "        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)\n"
    "    ):\n"
    "        # fallback: disabled-HMA hybrid specs via page-size unify.\n"
    "        unified_kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)\n"
    "        kv_cache_spec.clear()\n"
    "        kv_cache_spec.update(unified_kv_cache_spec)\n"
    "\n"
    "    if not (\n"
    "        is_kv_cache_spec_uniform(kv_cache_spec)\n"
    "        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)\n"
    "    ):\n"
    "        raise ValueError(\n"
    "            \"Hybrid KV cache manager is disabled but failed to \"\n"
    "            \"convert the KV cache specs to one unified type.\"\n"
    "        )\n"
)
new_replace = (
    "    if not (\n"
    "        is_kv_cache_spec_uniform(kv_cache_spec)\n"
    "        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)\n"
    "    ):\n"
    "        # fallback: disabled-HMA mixed specs continue to downstream page-size grouping.\n"
    "        return\n"
)
if new_marker not in text:
    if old_marker in text and old_replace in text:
        text = text.replace(old_replace, new_replace, 1)
    elif needle in text:
        text = text.replace(needle, new_replace, 1)
    else:
        raise RuntimeError(
            f"Could not find disabled-HMA hybrid KV fallback target in {kvu_path}"
        )
    kvu_path.write_text(text, encoding="utf-8")

print(f"patched_disabled_hma_hybrid_kv_fallback_offline={kvu_path}")
PY
fi

if [[ "${RUN_QWEN14B}" == "1" ]]; then
  echo "[step] generate model-aware hidden config (qwen14b)"
  run_python scripts/create_vllm_config_auto.py \
    --model "${QWEN14B_MODEL_ID}" \
    --layers auto4 \
    --output "${QWEN14B_CONFIG_OUT}" \
    > "${OUT_DIR}/create_cfg_qwen14b.log" 2>&1
else
  echo "[info] RUN_QWEN14B=0; skipping qwen14b hidden config generation"
fi

if [[ "${RUN_QWEN8B}" == "1" ]]; then
  echo "[step] generate model-aware hidden config (qwen8b)"
  run_python scripts/create_vllm_config_auto.py \
    --model "${QWEN8B_MODEL_ID}" \
    --layers auto4 \
    --output "${QWEN8B_CONFIG_OUT}" \
    > "${OUT_DIR}/create_cfg_qwen8b.log" 2>&1
else
  echo "[info] RUN_QWEN8B=0; skipping qwen8b hidden config generation"
fi

if [[ ( "${USE_QWEN35_VLLM_HIDDEN}" == "1" || "${USE_QWEN35_VLLM_OFFLINE}" == "1" ) && ( "${RUN_QWEN9B}" == "1" || "${RUN_QWEN27B}" == "1" ) ]]; then
  if [[ "${USE_QWEN35_COMPAT_MODEL}" == "1" ]]; then
    echo "[step] prepare qwen3.5 compat model mirrors"
    PREPARE_QWEN35_ARGS=()
    PREPARE_QWEN9_ARCH_ARGS=()
    PREPARE_QWEN27_ARCH_ARGS=()
    CREATE_CFG_QWEN9_ARGS=()
    CREATE_CFG_QWEN27_ARGS=()
    if [[ "${VLLM_RUNTIME_VERSION}" == 0.14.* ]]; then
      PREPARE_QWEN35_ARGS+=(--force-qwen3)
      CREATE_CFG_QWEN9_ARGS+=(--verifier-architectures Qwen3ForCausalLM)
      CREATE_CFG_QWEN27_ARGS+=(--verifier-architectures Qwen3ForCausalLM)
    else
      PREPARE_QWEN9_ARCH_ARGS+=(--target-architecture Qwen3_5ForCausalLM)
      PREPARE_QWEN27_ARCH_ARGS+=(--target-architecture Qwen3_5MoeForCausalLM)
      CREATE_CFG_QWEN9_ARGS+=(--verifier-architectures Qwen3_5ForCausalLM)
      CREATE_CFG_QWEN27_ARGS+=(--verifier-architectures Qwen3_5MoeForCausalLM)
    fi

    if [[ "${RUN_QWEN9B}" == "1" ]]; then
      run_python scripts/prepare_qwen35_compat_model.py \
        --model-id "${QWEN9B_MODEL_ID}" \
        --output-dir "${QWEN9B_COMPAT_MODEL_DIR}" \
        "${PREPARE_QWEN35_ARGS[@]}" \
        "${PREPARE_QWEN9_ARCH_ARGS[@]}" \
        > "${OUT_DIR}/prepare_qwen9b_compat.log" 2>&1

      run_python scripts/create_vllm_config_auto.py \
        --model "${QWEN9B_MODEL_ID}" \
        --layers auto4 \
        "${CREATE_CFG_QWEN9_ARGS[@]}" \
        --verifier-name-or-path "${QWEN9B_COMPAT_MODEL_DIR}" \
        --output "${QWEN9B_CONFIG_OUT}" \
        > "${OUT_DIR}/create_cfg_qwen9b.log" 2>&1
    else
      echo "[info] RUN_QWEN9B=0; skipping qwen3.5-9b compat prep/config"
    fi

    if [[ "${RUN_QWEN27B}" == "1" ]]; then
      run_python scripts/prepare_qwen35_compat_model.py \
        --model-id "${QWEN27B_MODEL_ID}" \
        --output-dir "${QWEN27B_COMPAT_MODEL_DIR}" \
        "${PREPARE_QWEN35_ARGS[@]}" \
        "${PREPARE_QWEN27_ARCH_ARGS[@]}" \
        > "${OUT_DIR}/prepare_qwen27b_compat.log" 2>&1

      run_python scripts/create_vllm_config_auto.py \
        --model "${QWEN27B_MODEL_ID}" \
        --layers auto4 \
        "${CREATE_CFG_QWEN27_ARGS[@]}" \
        --verifier-name-or-path "${QWEN27B_COMPAT_MODEL_DIR}" \
        --output "${QWEN27B_CONFIG_OUT}" \
        > "${OUT_DIR}/create_cfg_qwen27b.log" 2>&1
    else
      echo "[info] RUN_QWEN27B=0; skipping qwen3.5-27b compat prep/config"
    fi

    echo "[step] inject eagle aux layer ids into qwen3.5 compat model configs"
    QWEN9B_CONFIG_OUT="${QWEN9B_CONFIG_OUT}" \
    QWEN9B_COMPAT_MODEL_DIR="${QWEN9B_COMPAT_MODEL_DIR}" \
    QWEN27B_CONFIG_OUT="${QWEN27B_CONFIG_OUT}" \
    QWEN27B_COMPAT_MODEL_DIR="${QWEN27B_COMPAT_MODEL_DIR}" \
    python3 - <<'PY'
import json
import os
from pathlib import Path


def _inject(spec_cfg_dir: str, compat_dir: str) -> None:
    spec_cfg_path = Path(spec_cfg_dir) / "config.json"
    compat_cfg_path = Path(compat_dir) / "config.json"
    if not spec_cfg_path.exists() or not compat_cfg_path.exists():
        print(f"skip_inject_missing:{spec_cfg_path}:{compat_cfg_path}")
        return

    spec_cfg = json.loads(spec_cfg_path.read_text(encoding="utf-8"))
    compat_cfg = json.loads(compat_cfg_path.read_text(encoding="utf-8"))

    layer_ids = spec_cfg.get("eagle_aux_hidden_state_layer_ids") or []
    if not layer_ids:
        print(f"skip_inject_no_layers:{spec_cfg_path}")
        return

    compat_cfg["eagle_aux_hidden_state_layer_ids"] = layer_ids
    compat_cfg.setdefault("draft_vocab_size", compat_cfg.get("vocab_size"))
    compat_cfg_path.write_text(
        json.dumps(compat_cfg, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"injected_layers:{compat_cfg_path}:{layer_ids}")


_inject(os.environ["QWEN9B_CONFIG_OUT"], os.environ["QWEN9B_COMPAT_MODEL_DIR"])
_inject(os.environ["QWEN27B_CONFIG_OUT"], os.environ["QWEN27B_COMPAT_MODEL_DIR"])
PY
else
  CREATE_CFG_QWEN9_NATIVE_ARGS=()
  CREATE_CFG_QWEN27_NATIVE_ARGS=()
  if [[ "${USE_QWEN35_VLLM_HIDDEN}" == "1" && "${VLLM_RUNTIME_VERSION}" != 0.14.* ]]; then
    # Native Qwen3.5 HF configs advertise *ForConditionalGeneration, but
    # vLLM hidden-state extractor configs need the text-only verifier classes.
    CREATE_CFG_QWEN9_NATIVE_ARGS+=(--verifier-architectures Qwen3_5ForCausalLM)
    CREATE_CFG_QWEN27_NATIVE_ARGS+=(--verifier-architectures Qwen3_5MoeForCausalLM)
  fi

  if [[ "${USE_QWEN35_VLLM_HIDDEN}" == "1" && "${RUN_QWEN9B}" == "1" ]]; then
    run_python scripts/create_vllm_config_auto.py \
      --model "${QWEN9B_MODEL_ID}" \
      --layers auto4 \
      "${CREATE_CFG_QWEN9_NATIVE_ARGS[@]}" \
      --output "${QWEN9B_CONFIG_OUT}" \
      > "${OUT_DIR}/create_cfg_qwen9b.log" 2>&1
  else
    echo "[info] skipping qwen3.5-9b hidden config generation (backend=${QWEN35_BACKEND}, run=${RUN_QWEN9B})"
  fi

  if [[ "${USE_QWEN35_VLLM_HIDDEN}" == "1" && "${RUN_QWEN27B}" == "1" ]]; then
    run_python scripts/create_vllm_config_auto.py \
      --model "${QWEN27B_MODEL_ID}" \
      --layers auto4 \
      "${CREATE_CFG_QWEN27_NATIVE_ARGS[@]}" \
      --output "${QWEN27B_CONFIG_OUT}" \
      > "${OUT_DIR}/create_cfg_qwen27b.log" 2>&1
  else
    echo "[info] skipping qwen3.5-27b hidden config generation (backend=${QWEN35_BACKEND}, run=${RUN_QWEN27B})"
  fi
  fi
fi

if [[ "${USE_ANY_PLUGIN_RUN}" == "1" || ( "${USE_QWEN35_VLLM_OFFLINE}" == "1" && "${DO_QWEN35_PATCH}" == "1" ) ]]; then
if [[ "${USE_ANY_PLUGIN_RUN}" == "1" ]]; then
echo "[step] patch hidden connector plugin"
PLUGIN_DIR="$(python3 -c 'import vllm_hidden_states_extractor, os; print(os.path.dirname(vllm_hidden_states_extractor.__file__))')"
if [[ "${ALLOW_PRIMARY_PLUGIN_PATCH}" == "1" ]]; then
  echo "[warn] patching hidden plugin in place because ALLOW_PRIMARY_PLUGIN_PATCH=1"
else
  SHADOW_PLUGIN_ROOT="${TMP_ROOT}/py_plugin_shadow"
  rm -rf "${SHADOW_PLUGIN_ROOT}"
  mkdir -p "${SHADOW_PLUGIN_ROOT}"
  cp -a "${PLUGIN_DIR}" "${SHADOW_PLUGIN_ROOT}/"
  PLUGIN_DIR="${SHADOW_PLUGIN_ROOT}/$(basename "${PLUGIN_DIR}")"
  if [[ -n "${VLLM_PYTHONPATH_PREFIX}" ]]; then
    VLLM_PYTHONPATH_PREFIX="${SHADOW_PLUGIN_ROOT}:${VLLM_PYTHONPATH_PREFIX}"
  else
    VLLM_PYTHONPATH_PREFIX="${SHADOW_PLUGIN_ROOT}"
  fi
  echo "[warn] using shadow hidden plugin dir: ${PLUGIN_DIR}"
fi
cp utils/vllm_hidden_connector.py "${PLUGIN_DIR}/connector.py"
echo "[info] plugin_dir=${PLUGIN_DIR}"
else
  echo "[info] skipping hidden connector plugin patch (no plugin-backed benchmarks enabled)"
fi

VLLM_PATCH_PACKAGE_DIR=""
if [[ ( "${USE_QWEN35_VLLM_HIDDEN}" == "1" || "${USE_QWEN35_VLLM_OFFLINE}" == "1" ) && "${DO_QWEN35_PATCH}" == "1" && "${ALLOW_PRIMARY_PLUGIN_PATCH}" != "1" ]]; then
  BASE_VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
  SHADOW_VLLM_ROOT="${TMP_ROOT}/py_vllm_shadow"
  rm -rf "${SHADOW_VLLM_ROOT}"
  mkdir -p "${SHADOW_VLLM_ROOT}"
  cp -a "${BASE_VLLM_DIR}" "${SHADOW_VLLM_ROOT}/"
  VLLM_PATCH_PACKAGE_DIR="${SHADOW_VLLM_ROOT}/$(basename "${BASE_VLLM_DIR}")"
  if [[ -n "${VLLM_PYTHONPATH_PREFIX}" ]]; then
    VLLM_PYTHONPATH_PREFIX="${SHADOW_VLLM_ROOT}:${VLLM_PYTHONPATH_PREFIX}"
  else
    VLLM_PYTHONPATH_PREFIX="${SHADOW_VLLM_ROOT}"
  fi
  echo "[warn] using shadow vllm package dir: ${VLLM_PATCH_PACKAGE_DIR}"
fi
export VLLM_PATCH_PACKAGE_DIR

if [[ ( "${USE_QWEN35_VLLM_HIDDEN}" == "1" || "${USE_QWEN35_VLLM_OFFLINE}" == "1" ) && "${DO_QWEN35_PATCH}" == "1" ]]; then
if [[ "${VLLM_RUNTIME_VERSION}" == 0.14.* ]]; then
echo "[step] patch vLLM qwen2 loader for qwen3.5 wrapper prefixes"
python3 - <<'PY'
import os
import pathlib
import sys
import urllib.request

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    qwen2_path = pathlib.Path(shadow_dir) / "model_executor/models/qwen2.py"
else:
    qwen2_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/model_executor/models/qwen2.py"
    )
if not qwen2_path.exists():
    raise RuntimeError(f"Missing qwen2.py at expected path: {qwen2_path}")

# Start from a known-good baseline to avoid cumulative patch drift.
raw_url = "https://raw.githubusercontent.com/vllm-project/vllm/v0.14.0/vllm/model_executor/models/qwen2.py"
with urllib.request.urlopen(raw_url, timeout=30) as resp:
    baseline = resp.read().decode("utf-8")
qwen2_path.write_text(baseline, encoding="utf-8")

text = qwen2_path.read_text(encoding="utf-8")
needle = "                param = params_dict[name]\n"
inject = (
    "                # Compat for qwen3.5 wrapper naming variants.\n"
    "                if name not in params_dict:\n"
    "                    alts = []\n"
    "                    if name.startswith(\"language_model.model.\"):\n"
    "                        tail = name[len(\"language_model.model.\"):]\n"
    "                        alts.extend((f\"model.{tail}\", f\"language_model.{tail}\", f\"model.language_model.{tail}\", tail))\n"
    "                    elif name.startswith(\"language_model.\"):\n"
    "                        tail = name[len(\"language_model.\"):]\n"
    "                        alts.extend((f\"model.{tail}\", f\"language_model.model.{tail}\", f\"model.language_model.{tail}\", tail))\n"
    "                    elif name.startswith(\"model.language_model.\"):\n"
    "                        tail = name[len(\"model.language_model.\"):]\n"
    "                        alts.extend((f\"model.{tail}\", f\"language_model.{tail}\", f\"language_model.model.{tail}\", tail))\n"
    "                    elif name.startswith(\"model.\"):\n"
    "                        tail = name[len(\"model.\"):]\n"
    "                        alts.extend((f\"language_model.{tail}\", f\"language_model.model.{tail}\", f\"model.language_model.{tail}\", tail))\n"
    "                    else:\n"
    "                        alts.extend((f\"model.{name}\", f\"language_model.{name}\", f\"language_model.model.{name}\", f\"model.language_model.{name}\"))\n"
    "                    for alt in alts:\n"
    "                        if alt in params_dict:\n"
    "                            name = alt\n"
    "                            break\n"
    "                if name not in params_dict:\n"
    "                    continue\n"
    "                param = params_dict[name]\n"
)
if "Compat for qwen3.5 wrapper naming variants." not in text:
    if needle not in text:
        raise RuntimeError(f"Could not find param lookup target in {qwen2_path}")
    text = text.replace(needle, inject)

qwen2_path.write_text(text, encoding="utf-8")
print(f"patched_qwen2_loader={qwen2_path}")
PY

echo "[step] patch vLLM qwen3 loader to ignore qwen3.5 mtp heads"
python3 - <<'PY'
import os
import pathlib
import sys
import urllib.request

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    qwen3_path = pathlib.Path(shadow_dir) / "model_executor/models/qwen3.py"
else:
    qwen3_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/model_executor/models/qwen3.py"
    )
if not qwen3_path.exists():
    raise RuntimeError(f"Missing qwen3.py at expected path: {qwen3_path}")

# Reset to known baseline first.
raw_url = "https://raw.githubusercontent.com/vllm-project/vllm/v0.14.0/vllm/model_executor/models/qwen3.py"
with urllib.request.urlopen(raw_url, timeout=30) as resp:
    baseline = resp.read().decode("utf-8")
qwen3_path.write_text(baseline, encoding="utf-8")

text = qwen3_path.read_text(encoding="utf-8")
needle = (
    "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
    "        loader = AutoWeightsLoader(\n"
    "            self,\n"
    "            skip_prefixes=([\"lm_head.\"] if self.config.tie_word_embeddings else None),\n"
    "        )\n"
    "        return loader.load_weights(weights)\n"
)
replace = (
    "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
    "        skip_prefixes = []\n"
    "        if self.config.tie_word_embeddings:\n"
    "            skip_prefixes.append(\"lm_head.\")\n"
    "        # Qwen3.5 checkpoints include mtp.* heads that are not present in\n"
    "        # vLLM's Qwen3ForCausalLM implementation; ignore those tensors.\n"
    "        skip_prefixes.append(\"mtp.\")\n"
    "\n"
    "        def _normalize_name(name: str) -> str:\n"
    "            if name.startswith(\"model.language_model.model.\"):\n"
    "                return \"model.\" + name[len(\"model.language_model.model.\"):]\n"
    "            if name.startswith(\"model.language_model.\"):\n"
    "                return \"model.\" + name[len(\"model.language_model.\"):]\n"
    "            if name.startswith(\"language_model.model.\"):\n"
    "                return \"model.\" + name[len(\"language_model.model.\"):]\n"
    "            if name.startswith(\"language_model.\"):\n"
    "                tail = name[len(\"language_model.\"):]\n"
    "                if tail.startswith(\"model.\") or tail.startswith(\"lm_head.\"):\n"
    "                    return tail\n"
    "                return \"model.\" + tail\n"
    "            if name.startswith(\"model.model.\"):\n"
    "                return \"model.\" + name[len(\"model.model.\"):]\n"
    "            return name\n"
    "\n"
    "        def _iter_weights():\n"
    "            norm_debug = []\n"
    "            for name, tensor in weights:\n"
    "                norm_name = _normalize_name(name)\n"
    "                if len(norm_debug) < 20:\n"
    "                    norm_debug.append((name, norm_name))\n"
    "                if norm_name.startswith(\"mtp.\"):\n"
    "                    continue\n"
    "                yield norm_name, tensor\n"
    "            for src, dst in norm_debug:\n"
    "                print(f\"qwen35_norm_sample:{src}=>{dst}\")\n"
    "\n"
    "        loader = AutoWeightsLoader(\n"
    "            self,\n"
    "            skip_prefixes=(skip_prefixes or None),\n"
    "        )\n"
    "        return loader.load_weights(_iter_weights())\n"
)
if "Qwen3.5 checkpoints include mtp.* heads" not in text:
    if needle not in text:
        raise RuntimeError(f"Could not find qwen3 load_weights target in {qwen3_path}")
    text = text.replace(needle, replace, 1)

qwen3_path.write_text(text, encoding="utf-8")
print(f"patched_qwen3_loader={qwen3_path}")
PY
else
  echo "[info] using native qwen3_5 loaders on vLLM ${VLLM_RUNTIME_VERSION}; skipping legacy qwen2/qwen3 runtime patches"
  echo "[step] patch vLLM get_hf_text_config for dict text_config compatibility (vLLM ${VLLM_RUNTIME_VERSION})"
  python3 - <<'PY'
import os
import pathlib
import sys

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    cfg_path = pathlib.Path(shadow_dir) / "transformers_utils/config.py"
else:
    cfg_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/transformers_utils/config.py"
    )
if not cfg_path.exists():
    raise RuntimeError(f"Missing config.py at expected path: {cfg_path}")

text = cfg_path.read_text(encoding="utf-8")
if "dict text_config compatibility for speculative config" not in text:
    needle = (
        "    if text_config is not config and not hasattr(text_config, \"num_attention_heads\"):\n"
        "        raise ValueError(\n"
        "            \"The text_config extracted from the model config does not have \"\n"
        "            \"`num_attention_heads` attribute. This indicates a mismatch \"\n"
        "            \"between the model config and vLLM's expectations. Please \"\n"
        "            \"ensure that the model config is compatible with vLLM.\"\n"
        "        )\n"
    )
    replace = (
        "    if text_config is not config and not hasattr(text_config, \"num_attention_heads\"):\n"
        "        # dict text_config compatibility for speculative config\n"
        "        if isinstance(text_config, dict) and text_config.get(\"num_attention_heads\") is not None:\n"
        "            from types import SimpleNamespace\n"
        "            text_config = SimpleNamespace(**text_config)\n"
        "        else:\n"
        "            raise ValueError(\n"
        "                \"The text_config extracted from the model config does not have \"\n"
        "                \"`num_attention_heads` attribute. This indicates a mismatch \"\n"
        "                \"between the model config and vLLM's expectations. Please \"\n"
        "                \"ensure that the model config is compatible with vLLM.\"\n"
        "            )\n"
    )
    if needle not in text:
        raise RuntimeError(f"Could not find get_hf_text_config target in {cfg_path}")
    text = text.replace(needle, replace, 1)
    cfg_path.write_text(text, encoding="utf-8")

print(f"patched_hf_text_config={cfg_path}")
PY

  echo "[step] patch vLLM example hidden connector for HMA request-finished flow (vLLM ${VLLM_RUNTIME_VERSION})"
  python3 - <<'PY'
import os
import pathlib
import sys

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    conn_path = pathlib.Path(shadow_dir) / "distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py"
else:
    conn_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py"
    )
if not conn_path.exists():
    raise RuntimeError(f"Missing example_hidden_states_connector.py at expected path: {conn_path}")

text = conn_path.read_text(encoding="utf-8")
orig = text

import_needle = (
    "from vllm.distributed.kv_transfer.kv_connector.v1.base import (\n"
    "    KVConnectorBase_V1,\n"
    "    KVConnectorMetadata,\n"
    "    KVConnectorRole,\n"
    ")\n"
)
import_replace = (
    "from vllm.distributed.kv_transfer.kv_connector.v1.base import (\n"
    "    KVConnectorBase_V1,\n"
    "    KVConnectorMetadata,\n"
    "    KVConnectorRole,\n"
    "    SupportsHMA,\n"
    ")\n"
)
if "SupportsHMA" not in text:
    if import_needle not in text:
        raise RuntimeError(f"Could not find connector import target in {conn_path}")
    text = text.replace(import_needle, import_replace, 1)

if "class ExampleHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):" not in text:
    text = text.replace(
        "class ExampleHiddenStatesConnector(KVConnectorBase_V1):",
        "class ExampleHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):",
        1,
    )

if "def request_finished_all_groups(" not in text:
    needle = (
        "    def request_finished(\n"
        "        self,\n"
        "        request: \"Request\",\n"
        "        block_ids: list[int],\n"
        "    ) -> tuple[bool, dict[str, Any] | None]:\n"
    )
    replace = (
        "    def request_finished_all_groups(\n"
        "        self,\n"
        "        request: \"Request\",\n"
        "        block_ids: tuple[list[int], ...],\n"
        "    ) -> tuple[bool, dict[str, Any] | None]:\n"
        "        first_group = block_ids[0] if block_ids else []\n"
        "        return self.request_finished(request, first_group)\n\n"
        "    def request_finished(\n"
        "        self,\n"
        "        request: \"Request\",\n"
        "        block_ids: list[int],\n"
        "    ) -> tuple[bool, dict[str, Any] | None]:\n"
    )
    if needle not in text:
        raise RuntimeError(f"Could not find request_finished target in {conn_path}")
    text = text.replace(needle, replace, 1)

if text != orig:
    conn_path.write_text(text, encoding="utf-8")

print(f"patched_example_hidden_connector_hma={conn_path}")
PY

  echo "[step] patch vLLM example hidden connector to use attention-metadata slot mapping (vLLM ${VLLM_RUNTIME_VERSION})"
  python3 - <<'PY'
import os
import pathlib
import sys

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    conn_path = pathlib.Path(shadow_dir) / "distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py"
else:
    conn_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py"
    )
if not conn_path.exists():
    raise RuntimeError(f"Missing example_hidden_states_connector.py at expected path: {conn_path}")

text = conn_path.read_text(encoding="utf-8")
if "connector mode: all|last_token" not in text:
    init_needle = (
        "        self._storage_path = self._kv_transfer_config.get_from_extra_config(\n"
        "            \"shared_storage_path\", \"/tmp\"\n"
        "        )\n"
        "        self.cache_layers: list[str] = []  # set by self.register_kv_caches\n"
    )
    init_replace = (
        "        self._storage_path = self._kv_transfer_config.get_from_extra_config(\n"
        "            \"shared_storage_path\", \"/tmp\"\n"
        "        )\n"
        "        self._mode = self._kv_transfer_config.get_from_extra_config(\n"
        "            \"mode\", \"all\"\n"
        "        )\n"
        "        assert self._mode in (\"all\", \"last_token\"), (\n"
        "            f\"Invalid hidden connector mode: {self._mode}\"\n"
        "        )\n"
        "        # connector mode: all|last_token\n"
        "        self.cache_layers: list[str] = []  # set by self.register_kv_caches\n"
    )
    if init_needle not in text:
        raise RuntimeError(f"Could not find hidden connector __init__ target in {conn_path}")
    text = text.replace(init_needle, init_replace, 1)

if "Prefer the live per-step slot mapping from attention metadata" not in text:
    needle = (
        "        os.makedirs(self._storage_path, exist_ok=True)\n"
        "        for request in connector_metadata.requests:\n"
        "            hidden_states = extract_from_kv_cache(\n"
        "                kv_layer, request.slot_mapping, request.token_ids.shape[0]\n"
        "            )\n"
        "            tensors = {\n"
        "                \"hidden_states\": hidden_states.detach().cpu(),\n"
        "                \"token_ids\": request.token_ids.detach().cpu(),\n"
        "            }\n"
        "            safetensors.torch.save_file(tensors, request.filename)\n"
    )
    replace = (
        "        os.makedirs(self._storage_path, exist_ok=True)\n"
        "        attn_slot_mapping = getattr(attn_metadata, \"slot_mapping\", None)\n"
        "        cursor = 0\n"
        "        kv_capacity = int(kv_layer.shape[0] * kv_layer.shape[1])\n"
        "        for request in connector_metadata.requests:\n"
        "            req_num_tokens = request.token_ids.shape[0]\n"
        "            req_slot_mapping = request.slot_mapping[:req_num_tokens]\n"
        "            req_token_ids = request.token_ids[: req_slot_mapping.shape[0]]\n"
        "            # Prefer the live per-step slot mapping from attention metadata.\n"
        "            # Under HMA, reconstructing slot_mapping from request block IDs can\n"
        "            # point outside the current kv tensor and trip device-side asserts.\n"
        "            if isinstance(attn_slot_mapping, torch.Tensor):\n"
        "                candidate = attn_slot_mapping[cursor: cursor + req_slot_mapping.shape[0]]\n"
        "                cursor += req_slot_mapping.shape[0]\n"
        "                if candidate.numel() > 0:\n"
        "                    req_slot_mapping = candidate\n"
        "                    req_token_ids = req_token_ids[: candidate.shape[0]]\n"
        "            valid_mask = (req_slot_mapping >= 0) & (req_slot_mapping < kv_capacity)\n"
        "            if valid_mask.numel() == 0 or not bool(valid_mask.any().item()):\n"
        "                logger.warning(\n"
        "                    \"No valid slot mappings for req %s (kv_capacity=%d, num_slots=%d)\",\n"
        "                    getattr(request, \"req_id\", \"<unknown>\"),\n"
        "                    kv_capacity,\n"
        "                    int(req_slot_mapping.numel()),\n"
        "                )\n"
        "                continue\n"
        "            if not bool(valid_mask.all().item()):\n"
        "                num_dropped = int((~valid_mask).sum().item())\n"
        "                slot_min = int(req_slot_mapping.min().item())\n"
        "                slot_max = int(req_slot_mapping.max().item())\n"
        "                logger.warning(\n"
        "                    \"Dropping %d invalid slot mappings for req %s (kv_capacity=%d, slot_min=%d, slot_max=%d)\",\n"
        "                    num_dropped,\n"
        "                    getattr(request, \"req_id\", \"<unknown>\"),\n"
        "                    kv_capacity,\n"
        "                    slot_min,\n"
        "                    slot_max,\n"
        "                )\n"
        "                req_slot_mapping = req_slot_mapping[valid_mask]\n"
        "                req_token_ids = req_token_ids[valid_mask.detach().cpu()]\n"
        "            if self._mode == \"last_token\" and req_slot_mapping.numel() > 1:\n"
        "                req_slot_mapping = req_slot_mapping[-1:]\n"
        "                req_token_ids = req_token_ids[-1:]\n"
        "            req_extract_tokens = req_slot_mapping.shape[0]\n"
        "            hidden_states = extract_from_kv_cache(\n"
        "                kv_layer, req_slot_mapping, req_extract_tokens\n"
        "            )\n"
        "            tensors = {\n"
        "                \"hidden_states\": hidden_states.detach().cpu(),\n"
        "                \"token_ids\": req_token_ids[:req_extract_tokens].detach().cpu(),\n"
        "            }\n"
        "            safetensors.torch.save_file(tensors, request.filename)\n"
    )
    if needle not in text:
        raise RuntimeError(f"Could not find save_kv_layer slot-mapping target in {conn_path}")
    text = text.replace(needle, replace, 1)
    conn_path.write_text(text, encoding="utf-8")

print(f"patched_example_hidden_connector_slot_mapping={conn_path}")
PY

  echo "[step] patch vLLM extract_hidden_states layer-id hydration (vLLM ${VLLM_RUNTIME_VERSION})"
  python3 - <<'PY'
import os
import pathlib
import sys

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    spec_path = pathlib.Path(shadow_dir) / "config/speculative.py"
else:
    spec_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/config/speculative.py"
    )
if not spec_path.exists():
    raise RuntimeError(f"Missing speculative.py at expected path: {spec_path}")

text = spec_path.read_text(encoding="utf-8")
if "extract_hidden_states layer-id hydration from speculator config" not in text:
    needle = (
        "            else:\n"
        "                hf_config = {}\n"
        "\n"
        "            self.draft_model_config = copy.copy(self.target_model_config)\n"
    )
    replace = (
        "            else:\n"
        "                hf_config = {}\n"
        "\n"
        "            # extract_hidden_states layer-id hydration from speculator config\n"
        "            if not hf_config and isinstance(self.model, str) and self.model not in (\"\", \"extract_hidden_states\"):\n"
        "                try:\n"
        "                    from transformers import PretrainedConfig\n"
        "                    spec_cfg_dict, _ = PretrainedConfig.get_config_dict(\n"
        "                        self.model,\n"
        "                        trust_remote_code=self.target_model_config.trust_remote_code,\n"
        "                        revision=self.revision,\n"
        "                    )\n"
        "                    if isinstance(spec_cfg_dict, dict):\n"
        "                        layer_ids = spec_cfg_dict.get(\"eagle_aux_hidden_state_layer_ids\")\n"
        "                        if layer_ids:\n"
        "                            hf_config[\"eagle_aux_hidden_state_layer_ids\"] = layer_ids\n"
        "                except Exception:\n"
        "                    pass\n"
        "\n"
        "            self.draft_model_config = copy.copy(self.target_model_config)\n"
    )
    if needle not in text:
        raise RuntimeError(f"Could not find extract_hidden_states hydration target in {spec_path}")
    text = text.replace(needle, replace, 1)
    spec_path.write_text(text, encoding="utf-8")

print(f"patched_extract_hidden_states_hydration={spec_path}")
PY

  echo "[step] patch vLLM extract_hidden_states layer-id env fallback (vLLM ${VLLM_RUNTIME_VERSION})"
  python3 - <<'PY'
import os
import pathlib
import sys

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    ehs_path = pathlib.Path(shadow_dir) / "v1/spec_decode/extract_hidden_states.py"
else:
    ehs_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/v1/spec_decode/extract_hidden_states.py"
    )
if not ehs_path.exists():
    raise RuntimeError(f"Missing extract_hidden_states.py at expected path: {ehs_path}")

text = ehs_path.read_text(encoding="utf-8")
if "VLLM_EAGLE_AUX_HIDDEN_STATE_LAYER_IDS" not in text:
    needle = (
        "        layer_ids = getattr(self.hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
        "        if not layer_ids:\n"
        "            raise ValueError(\n"
        "                \"eagle_aux_hidden_state_layer_ids must be set in the draft \"\n"
        "                \"model config for extract_hidden_states method\"\n"
        "            )\n"
    )
    replace = (
        "        layer_ids = getattr(self.hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
        "        if not layer_ids:\n"
        "            raw_layer_ids = __import__(\"os\").environ.get(\"VLLM_EAGLE_AUX_HIDDEN_STATE_LAYER_IDS\", \"\").strip()\n"
        "            if raw_layer_ids:\n"
        "                try:\n"
        "                    layer_ids = [int(x.strip()) for x in raw_layer_ids.split(\",\") if x.strip()]\n"
        "                except ValueError:\n"
        "                    layer_ids = None\n"
        "                if layer_ids:\n"
        "                    setattr(self.hf_config, \"eagle_aux_hidden_state_layer_ids\", layer_ids)\n"
        "        if not layer_ids:\n"
        "            raise ValueError(\n"
        "                \"eagle_aux_hidden_state_layer_ids must be set in the draft \"\n"
        "                \"model config for extract_hidden_states method\"\n"
        "            )\n"
    )
    if needle not in text:
        raise RuntimeError(f"Could not find extract_hidden_states layer-id target in {ehs_path}")
    text = text.replace(needle, replace, 1)
    ehs_path.write_text(text, encoding="utf-8")

print(f"patched_extract_hidden_states_env_fallback={ehs_path}")
PY

  echo "[step] patch vLLM gpu_model_runner tuple-unpack compatibility (vLLM ${VLLM_RUNTIME_VERSION})"
  python3 - <<'PY'
import os
import pathlib
import sys

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    gmr_path = pathlib.Path(shadow_dir) / "v1/worker/gpu_model_runner.py"
else:
    gmr_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py"
    )
if not gmr_path.exists():
    raise RuntimeError(f"Missing gpu_model_runner.py at expected path: {gmr_path}")

text = gmr_path.read_text(encoding="utf-8")

needle_main = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                # True when EAGLE 3 is used.\n"
    "                hidden_states, aux_hidden_states = model_output\n"
    "            else:\n"
)
replace_main = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                # True when EAGLE 3 is used.\n"
    "                if isinstance(model_output, tuple):\n"
    "                    hidden_states = model_output[0]\n"
    "                    aux_hidden_states = model_output[1] if len(model_output) > 1 else None\n"
    "                else:\n"
    "                    hidden_states = model_output\n"
    "                    aux_hidden_states = None\n"
    "            else:\n"
)
if "aux_hidden_states = model_output[1] if len(model_output) > 1 else None" not in text:
    if needle_main not in text:
        raise RuntimeError(f"Could not find main tuple-unpack target in {gmr_path}")
    text = text.replace(needle_main, replace_main, 1)

needle_dummy = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                hidden_states, _ = outputs\n"
    "            else:\n"
)
replace_dummy = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                if isinstance(outputs, tuple):\n"
    "                    hidden_states = outputs[0]\n"
    "                else:\n"
    "                    hidden_states = outputs\n"
    "            else:\n"
)
if "if isinstance(outputs, tuple):\n                    hidden_states = outputs[0]" not in text:
    if needle_dummy not in text:
        raise RuntimeError(f"Could not find dummy tuple-unpack target in {gmr_path}")
    text = text.replace(needle_dummy, replace_dummy, 1)

if "target_hidden_states = [base_hidden for _ in range(num_aux_layers)]" not in text:
    lines = text.splitlines(keepends=True)
    marker_idx = next(
        (i for i, line in enumerate(lines)
         if "aux_hidden_states are required when using `extract_hidden_states`" in line),
        None,
    )
    if marker_idx is None:
        raise RuntimeError(f"Could not find extract_hidden_states error marker in {gmr_path}")

    start_idx = marker_idx
    while start_idx >= 0 and "if not self.use_aux_hidden_state_outputs" not in lines[start_idx]:
        start_idx -= 1
    if start_idx < 0:
        raise RuntimeError(f"Could not find extract_hidden_states guard start in {gmr_path}")

    end_idx = marker_idx
    while end_idx < len(lines) and "target_hidden_states =" not in lines[end_idx]:
        end_idx += 1
    if end_idx >= len(lines):
        raise RuntimeError(f"Could not find extract_hidden_states target_hidden_states line in {gmr_path}")

    indent = lines[start_idx][: len(lines[start_idx]) - len(lines[start_idx].lstrip())]
    replacement_lines = [
        f"{indent}if not self.use_aux_hidden_state_outputs:\n",
        f"{indent}    raise ValueError(\n",
        f"{indent}        \"aux_hidden_states are required when using `extract_hidden_states`\"\n",
        f"{indent}    )\n",
        f"{indent}if aux_hidden_states is None:\n",
        f"{indent}    # Runtime compatibility fallback: some qwen3.5 + plugin\n",
        f"{indent}    # paths return only the main hidden state tensor.\n",
        f"{indent}    # Mirror it across expected aux layers to avoid engine death.\n",
        f"{indent}    layer_ids = getattr(\n",
        f"{indent}        self.speculative_config.draft_model_config.hf_config,\n",
        f"{indent}        \"eagle_aux_hidden_state_layer_ids\",\n",
        f"{indent}        None,\n",
        f"{indent}    )\n",
        f"{indent}    num_aux_layers = len(layer_ids) if layer_ids else 1\n",
        f"{indent}    base_hidden = hidden_states[:num_scheduled_tokens]\n",
        f"{indent}    target_hidden_states = [base_hidden for _ in range(num_aux_layers)]\n",
        f"{indent}else:\n",
        f"{indent}    target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]\n",
    ]
    lines[start_idx:end_idx + 1] = replacement_lines
    text = "".join(lines)

gmr_path.write_text(text, encoding="utf-8")
print(f"patched_gpu_model_runner_tuple_unpack={gmr_path}")
PY

  echo "[step] patch vLLM scheduler request-finished compatibility for multi-group hidden connector (vLLM ${VLLM_RUNTIME_VERSION})"
  python3 - <<'PY'
import os
import pathlib
import sys

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    sched_path = pathlib.Path(shadow_dir) / "v1/core/sched/scheduler.py"
else:
    sched_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py"
    )
if not sched_path.exists():
    raise RuntimeError(f"Missing scheduler.py at expected path: {sched_path}")

text = sched_path.read_text(encoding="utf-8")
needle = (
    "        if not isinstance(self.connector, SupportsHMA):\n"
    "            # NOTE(Kuntai): We should deprecate this code path after we enforce\n"
    "            # all connectors to support HMA.\n"
    "            # Hybrid memory allocator should be already turned off for this\n"
    "            # code path, but let's double-check here.\n"
    "            assert len(self.kv_cache_config.kv_cache_groups) == 1\n"
    "            return self.connector.request_finished(request, block_ids[0])\n"
    "\n"
    "        return self.connector.request_finished_all_groups(request, block_ids)\n"
)
replace = (
    "        if not isinstance(self.connector, SupportsHMA):\n"
    "            # NOTE(Kuntai): We should deprecate this code path after we enforce\n"
    "            # all connectors to support HMA.\n"
    "            # Compatibility fallback for hidden-state connectors on hybrid\n"
    "            # models: if the connector exposes request_finished_all_groups,\n"
    "            # prefer that path even without SupportsHMA registration.\n"
    "            if len(self.kv_cache_config.kv_cache_groups) != 1 and hasattr(\n"
    "                self.connector, \"request_finished_all_groups\"\n"
    "            ):\n"
    "                return self.connector.request_finished_all_groups(request, block_ids)\n"
    "            # Hybrid memory allocator should be already turned off for this\n"
    "            # code path, but let's double-check here.\n"
    "            assert len(self.kv_cache_config.kv_cache_groups) == 1\n"
    "            return self.connector.request_finished(request, block_ids[0])\n"
    "\n"
    "        return self.connector.request_finished_all_groups(request, block_ids)\n"
)
if "Compatibility fallback for hidden-state connectors on hybrid" not in text:
    if needle not in text:
        raise RuntimeError(f"Could not find scheduler connector-finished target in {sched_path}")
    text = text.replace(needle, replace, 1)
    sched_path.write_text(text, encoding="utf-8")

print(f"patched_scheduler_request_finished_hma={sched_path}")
PY

  echo "[step] patch vLLM connector HMA gate for hidden-states connector (vLLM ${VLLM_RUNTIME_VERSION})"
  python3 - <<'PY'
import os
import pathlib
import sys

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    factory_path = pathlib.Path(shadow_dir) / "distributed/kv_transfer/kv_connector/factory.py"
else:
    factory_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/distributed/kv_transfer/kv_connector/factory.py"
    )
if not factory_path.exists():
    raise RuntimeError(f"Missing factory.py at expected path: {factory_path}")

text = factory_path.read_text(encoding="utf-8")
if "Bypassing HMA support check for ExampleHiddenStatesConnector" not in text:
    needle = (
        "        if hma_enabled and not supports_hma(connector_cls):\n"
        "            raise ValueError(\n"
        "                f\"Connector {connector_cls.__name__} does not support HMA but \"\n"
        "                f\"HMA is enabled. Please set `--disable-hybrid-kv-cache-manager`.\"\n"
        "            )\n"
    )
    replace = (
        "        if hma_enabled and not supports_hma(connector_cls):\n"
        "            if connector_cls.__name__ == \"ExampleHiddenStatesConnector\":\n"
        "                logger.warning(\n"
        "                    \"Bypassing HMA support check for ExampleHiddenStatesConnector\"\n"
        "                )\n"
        "            else:\n"
        "                raise ValueError(\n"
        "                    f\"Connector {connector_cls.__name__} does not support HMA but \"\n"
        "                    f\"HMA is enabled. Please set `--disable-hybrid-kv-cache-manager`.\"\n"
        "                )\n"
    )
    if needle not in text:
        raise RuntimeError(f"Could not find connector HMA gate target in {factory_path}")
    text = text.replace(needle, replace, 1)
    factory_path.write_text(text, encoding="utf-8")

print(f"patched_kv_connector_factory_hma_gate={factory_path}")
PY

  echo "[step] patch vLLM example hidden connector cached block handling (vLLM ${VLLM_RUNTIME_VERSION})"
  python3 - <<'PY'
import os
import pathlib
import sys

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    conn_path = pathlib.Path(shadow_dir) / "distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py"
else:
    conn_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py"
    )
if not conn_path.exists():
    raise RuntimeError(f"Missing example_hidden_states_connector.py at expected path: {conn_path}")

text = conn_path.read_text(encoding="utf-8")
if "new_block_ids can be None under some scheduler paths" not in text:
    full_needle = (
        "            new_block_ids = cached_reqs.new_block_ids[i]\n"
        "            cached_req = self._active_requests[req_id]\n"
        "            req_block_ids = self._req_blocks[req_id]\n"
        "            assert new_block_ids is not None\n"
        "            block_ids = new_block_ids[0]\n"
        "            req_block_ids.extend(block_ids)\n"
    )
    full_replace = (
        "            new_block_ids = cached_reqs.new_block_ids[i]\n"
        "            cached_req = self._active_requests[req_id]\n"
        "            req_block_ids = self._req_blocks[req_id]\n"
        "            # new_block_ids can be None under some scheduler paths.\n"
        "            # Reuse existing tracked block ids instead of asserting.\n"
        "            if new_block_ids is None:\n"
        "                block_ids = []\n"
        "            else:\n"
        "                block_ids = new_block_ids[0] if len(new_block_ids) > 0 else []\n"
        "            req_block_ids.extend(block_ids)\n"
    )
    narrow_needle = (
        "            assert new_block_ids is not None\n"
        "            block_ids = new_block_ids[0]\n"
    )
    narrow_replace = (
        "            # new_block_ids can be None under some scheduler paths.\n"
        "            # Reuse existing tracked block ids instead of asserting.\n"
        "            if new_block_ids is None:\n"
        "                block_ids = []\n"
        "            else:\n"
        "                block_ids = new_block_ids[0] if len(new_block_ids) > 0 else []\n"
    )
    if full_needle in text:
        text = text.replace(full_needle, full_replace, 1)
    elif narrow_needle in text:
        text = text.replace(narrow_needle, narrow_replace, 1)
    else:
        raise RuntimeError(
            f"Could not find cached block-id patch target in {conn_path}"
        )
    conn_path.write_text(text, encoding="utf-8")

print(f"patched_example_hidden_connector_cached_blocks={conn_path}")
PY

  echo "[step] patch vLLM kv page-size unification fallback (vLLM ${VLLM_RUNTIME_VERSION})"
  python3 - <<'PY'
import os
import pathlib
import sys

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    kvu_path = pathlib.Path(shadow_dir) / "v1/core/kv_cache_utils.py"
else:
    kvu_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/v1/core/kv_cache_utils.py"
    )
if not kvu_path.exists():
    raise RuntimeError(f"Missing kv_cache_utils.py at expected path: {kvu_path}")

text = kvu_path.read_text(encoding="utf-8")
if "fallback: unify via page_size_padded" not in text:
    needle = (
        "        else:\n"
        "            layer_page_size = layer_spec.page_size_bytes\n"
        "            if max_page_size % layer_page_size != 0:\n"
        "                raise NotImplementedError(\n"
        "                    \"The page size of the layer is not divisible by the \"\n"
        "                    \"maximum page size. Cannot unify by adjusting block_size.\"\n"
        "                )\n"
        "            ratio = max_page_size // layer_page_size\n"
        "            new_block_size = layer_spec.block_size * ratio\n"
        "            new_spec = replace(layer_spec, block_size=new_block_size)\n"
        "            assert new_spec.page_size_bytes == max_page_size\n"
        "            new_kv_cache_spec[layer_name] = new_spec\n"
    )
    replace = (
        "        else:\n"
        "            layer_page_size = layer_spec.page_size_bytes\n"
        "            new_spec = None\n"
        "            # First try the original block-size scaling path.\n"
        "            if max_page_size % layer_page_size == 0:\n"
        "                ratio = max_page_size // layer_page_size\n"
        "                new_block_size = layer_spec.block_size * ratio\n"
        "                cand = replace(layer_spec, block_size=new_block_size)\n"
        "                if cand.page_size_bytes == max_page_size:\n"
        "                    new_spec = cand\n"
        "\n"
        "            if new_spec is None:\n"
        "                # fallback: unify via page_size_padded for specs where\n"
        "                # page_size is not linear in block_size (e.g. mamba/mixed).\n"
        "                if hasattr(layer_spec, \"page_size_padded\"):\n"
        "                    cand = replace(layer_spec, page_size_padded=max_page_size)\n"
        "                    if cand.page_size_bytes == max_page_size:\n"
        "                        new_spec = cand\n"
                "\n"
        "            if new_spec is None:\n"
        "                raise NotImplementedError(\n"
        "                    \"The page size of the layer is not divisible by the \"\n"
        "                    \"maximum page size and page_size_padded fallback failed.\"\n"
        "                )\n"
        "\n"
        "            new_kv_cache_spec[layer_name] = new_spec\n"
    )
    if needle not in text:
        raise RuntimeError(f"Could not find kv page-size unify target in {kvu_path}")
    text = text.replace(needle, replace, 1)
    kvu_path.write_text(text, encoding="utf-8")

print(f"patched_kv_cache_unify_fallback={kvu_path}")
PY

  echo "[step] patch vLLM disabled-HMA hybrid KV fallback (vLLM ${VLLM_RUNTIME_VERSION})"
  python3 - <<'PY'
import os
import pathlib
import sys

shadow_dir = os.environ.get("VLLM_PATCH_PACKAGE_DIR", "")
if shadow_dir:
    kvu_path = pathlib.Path(shadow_dir) / "v1/core/kv_cache_utils.py"
else:
    kvu_path = (
        pathlib.Path(sys.prefix)
        / "lib/python3.10/site-packages/vllm/v1/core/kv_cache_utils.py"
    )
if not kvu_path.exists():
    raise RuntimeError(f"Missing kv_cache_utils.py at expected path: {kvu_path}")

text = kvu_path.read_text(encoding="utf-8")
new_marker = "fallback: disabled-HMA mixed specs continue to downstream page-size grouping"
old_marker = "fallback: disabled-HMA hybrid specs via page-size unify"
needle = (
    "    if not (\n"
    "        is_kv_cache_spec_uniform(kv_cache_spec)\n"
    "        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)\n"
    "    ):\n"
    "        raise ValueError(\n"
    "            \"Hybrid KV cache manager is disabled but failed to \"\n"
    "            \"convert the KV cache specs to one unified type.\"\n"
    "        )\n"
)
old_replace = (
    "    if not (\n"
    "        is_kv_cache_spec_uniform(kv_cache_spec)\n"
    "        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)\n"
    "    ):\n"
    "        # fallback: disabled-HMA hybrid specs via page-size unify.\n"
    "        unified_kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)\n"
    "        kv_cache_spec.clear()\n"
    "        kv_cache_spec.update(unified_kv_cache_spec)\n"
    "\n"
    "    if not (\n"
    "        is_kv_cache_spec_uniform(kv_cache_spec)\n"
    "        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)\n"
    "    ):\n"
    "        raise ValueError(\n"
    "            \"Hybrid KV cache manager is disabled but failed to \"\n"
    "            \"convert the KV cache specs to one unified type.\"\n"
    "        )\n"
)
new_replace = (
    "    if not (\n"
    "        is_kv_cache_spec_uniform(kv_cache_spec)\n"
    "        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)\n"
    "    ):\n"
    "        # fallback: disabled-HMA mixed specs continue to downstream page-size grouping.\n"
    "        return\n"
)
if new_marker not in text:
    if old_marker in text and old_replace in text:
        text = text.replace(old_replace, new_replace, 1)
    elif needle in text:
        text = text.replace(needle, new_replace, 1)
    else:
        raise RuntimeError(
            f"Could not find disabled-HMA hybrid KV fallback target in {kvu_path}"
        )
    kvu_path.write_text(text, encoding="utf-8")

print(f"patched_disabled_hma_hybrid_kv_fallback={kvu_path}")
PY
fi
else
  echo "[info] qwen3.5 runtime patching disabled (backend=${QWEN35_BACKEND}, DO_QWEN35_PATCH=${DO_QWEN35_PATCH})"
fi

if [[ -n "${HF_OVERLAY_PYTHONPATH}" ]]; then
  echo "[step] patch vLLM speculators parser for transformers>=5 compatibility"
  python3 - <<'PY'
import inspect
import pathlib

import vllm.transformers_utils.configs.speculators.base as spec_base

base_path = pathlib.Path(inspect.getsourcefile(spec_base))
text = base_path.read_text(encoding="utf-8")
needle = "return cls(**vllm_config)"
inject = 'vllm_config.pop("method", None)\n        return cls(**vllm_config)'
if inject not in text:
    if needle not in text:
        raise RuntimeError(f"Could not find patch target in {base_path}")
    text = text.replace(needle, inject, 1)
    base_path.write_text(text, encoding="utf-8")
print(f"patched_speculators_base={base_path}")
PY
fi
else
  echo "[info] skipping vLLM plugin/runtime patch setup (no plugin-backed benchmarks and no offline qwen3.5 runtime patch requested)"
fi

wait_vllm_ready() {
  local port="$1"
  local pid="${2:-}"
  local max_wait_s="${3:-600}"
  local poll_s=2
  local tries=$((max_wait_s / poll_s))
  if [[ "${tries}" -le 0 ]]; then
    tries=1
  fi
  for _ in $(seq 1 "${tries}"); do
    if curl -sSf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
      return 2
    fi
    sleep "${poll_s}"
  done
  return 1
}

run_qwen_bench() {
  local label="$1"
  local config_dir="$2"
  local served_model="$3"
  local tp_size="$4"
  local port="$5"
  local gpu_util="$6"

  local hidden_dir="${TMP_ROOT}/hidden_${label}"
  local vllm_log="${OUT_DIR}/vllm_${label}.log"
  local bench_out="${OUT_DIR}/${label}"

  mkdir -p "${hidden_dir}" "${bench_out}"
  rm -f "${hidden_dir}"/*.safetensors || true

  local eagle_layer_ids=""
  if [[ -f "${config_dir}/config.json" ]]; then
    eagle_layer_ids="$(
      CONFIG_DIR="${config_dir}" python3 - <<'PY'
import json
import os
from pathlib import Path

cfg_path = Path(os.environ["CONFIG_DIR"]) / "config.json"
try:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    vals = [int(v) for v in (cfg.get("eagle_aux_hidden_state_layer_ids") or [])]
    print(",".join(str(v) for v in vals), end="")
except Exception:
    pass
PY
    )"
  fi
  if [[ -n "${eagle_layer_ids}" ]]; then
    echo "[info] ${label} eagle_aux_hidden_state_layer_ids=${eagle_layer_ids}"
    export VLLM_EAGLE_AUX_HIDDEN_STATE_LAYER_IDS="${eagle_layer_ids}"
  else
    unset VLLM_EAGLE_AUX_HIDDEN_STATE_LAYER_IDS || true
  fi

  echo "[step] start vLLM (${label})"
  EXTRA_ARGS=()
  if [[ "${SAFE_MODE}" == "1" ]]; then
    EXTRA_ARGS+=(--enforce-eager)
  fi
  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    EXTRA_ARGS+=(--trust-remote-code)
  fi
  if [[ "${VLLM_RUNTIME_VERSION}" != 0.14.* ]]; then
    local enable_hma_flag=0
    if [[ "${ENABLE_HYBRID_KV_CACHE_MANAGER}" == "1" ]]; then
      enable_hma_flag=1
    elif [[ "${ENABLE_HYBRID_KV_CACHE_MANAGER}" == "auto" && "${label}" == qwen3_5_* ]]; then
      enable_hma_flag=1
    fi
    if [[ "${enable_hma_flag}" == "1" ]]; then
      EXTRA_ARGS+=(--no-disable-hybrid-kv-cache-manager)
    fi
  fi

  if [[ -n "${VLLM_PYTHONPATH_PREFIX}" ]]; then
    PYTHONPATH="${VLLM_PYTHONPATH_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}" \
    CRAFTAX_QWEN35_ALIAS_DEBUG=1 \
    vllm serve "${config_dir}" \
      --host 127.0.0.1 \
      --port "${port}" \
      --served-model-name "${served_model}" \
      --tensor-parallel-size "${tp_size}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${gpu_util}" \
      --kv-transfer-config "{\"kv_connector\":\"ExampleHiddenStatesConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"${hidden_dir}\",\"mode\":\"last_token\"}}" \
      "${EXTRA_ARGS[@]}" \
      > "${vllm_log}" 2>&1 &
  else
    vllm serve "${config_dir}" \
      --host 127.0.0.1 \
      --port "${port}" \
      --served-model-name "${served_model}" \
      --tensor-parallel-size "${tp_size}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${gpu_util}" \
      --kv-transfer-config "{\"kv_connector\":\"ExampleHiddenStatesConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"${hidden_dir}\",\"mode\":\"last_token\"}}" \
      "${EXTRA_ARGS[@]}" \
      > "${vllm_log}" 2>&1 &
  fi
  local vllm_pid=$!

  cleanup_vllm() {
    kill "${vllm_pid}" >/dev/null 2>&1 || true
    wait "${vllm_pid}" 2>/dev/null || true
  }
  trap cleanup_vllm RETURN

  if ! wait_vllm_ready "${port}" "${vllm_pid}" 900; then
    echo "[error] vLLM (${label}) failed readiness"
    tail -n 200 "${vllm_log}" || true
    return 1
  fi
  echo "[info] vLLM ready (${label}) port=${port}"

  run_python scripts/benchmark_future_imagination_latency.py \
    --provider openai_compatible \
    --base-url "http://127.0.0.1:${port}" \
    --model "${served_model}" \
    --prompt-dir-state "${PROMPT_STATE_DIR}" \
    --prompt-dir-history "${PROMPT_HISTORY_DIR}" \
    --benchmark-mode "${BENCHMARK_MODE}" \
    --warmup-prompts-per-variant "${WARMUP_PROMPTS_PER_VARIANT}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --repeats "${REPEATS}" \
    --timeout-s "${REQUEST_TIMEOUT_S}" \
    --hidden-states-path "${hidden_dir}" \
    --hidden-target-layer -1 \
    --delete-hidden-after-read \
    --output-dir "${bench_out}" \
    --run-label "${label}"

  cleanup_vllm
  trap - RETURN
  unset VLLM_EAGLE_AUX_HIDDEN_STATE_LAYER_IDS || true
}

run_qwen_hf_bench() {
  local label="$1"
  local model_id="$2"
  local bench_out="${OUT_DIR}/${label}"
  local hf_cache_dir="${QWEN35_HF_CACHE_ROOT}/${label}"

  mkdir -p "${bench_out}" "${hf_cache_dir}"
  echo "[step] run HF local benchmark (${label}) model=${model_id}"
  echo "[info] hf_cache_dir=${hf_cache_dir}"

  HF_ARGS=()
  if [[ "${QWEN35_HF_TRUST_REMOTE_CODE}" == "1" ]]; then
    HF_ARGS+=(--hf-trust-remote-code)
  fi
  if [[ -n "${QWEN35_HF_ATTN_IMPL}" ]]; then
    HF_ARGS+=(--hf-attn-implementation "${QWEN35_HF_ATTN_IMPL}")
  fi

  HF_HOME="${hf_cache_dir}" \
  HUGGINGFACE_HUB_CACHE="${hf_cache_dir}" \
  TRANSFORMERS_CACHE="${hf_cache_dir}" \
  run_hf_python scripts/benchmark_future_imagination_latency.py \
    --provider hf_local \
    --model "${model_id}" \
    --hf-model-id "${model_id}" \
    --hf-device-map "${QWEN35_HF_DEVICE_MAP}" \
    --hf-dtype "${QWEN35_HF_DTYPE}" \
    "${HF_ARGS[@]}" \
    --prompt-dir-state "${PROMPT_STATE_DIR}" \
    --prompt-dir-history "${PROMPT_HISTORY_DIR}" \
    --benchmark-mode "${BENCHMARK_MODE}" \
    --warmup-prompts-per-variant "${WARMUP_PROMPTS_PER_VARIANT}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --repeats "${REPEATS}" \
    --timeout-s "${REQUEST_TIMEOUT_S}" \
    --hidden-target-layer -1 \
    --output-dir "${bench_out}" \
    --run-label "${label}"
}

run_qwen_vllm_offline_bench() {
  local label="$1"
  local model_id="$2"
  local tp_size="$3"
  local gpu_util="$4"
  local bench_out="${OUT_DIR}/${label}"

  mkdir -p "${bench_out}"
  echo "[step] run native vLLM offline benchmark (${label}) model=${model_id}"

  VLLM_OFFLINE_ARGS=()
  if [[ "${QWEN35_VLLM_OFFLINE_TRUST_REMOTE_CODE}" == "1" ]]; then
    VLLM_OFFLINE_ARGS+=(--vllm-trust-remote-code)
  fi
  if [[ "${QWEN35_VLLM_OFFLINE_ENFORCE_EAGER}" == "1" ]]; then
    VLLM_OFFLINE_ARGS+=(--vllm-enforce-eager)
  fi
  if [[ "${QWEN35_VLLM_OFFLINE_PREFIX_CACHING}" == "1" ]]; then
    VLLM_OFFLINE_ARGS+=(--vllm-enable-prefix-caching)
  fi
  if [[ "${QWEN35_VLLM_OFFLINE_CHUNKED_PREFILL}" == "1" ]]; then
    VLLM_OFFLINE_ARGS+=(--vllm-enable-chunked-prefill)
  fi
  if [[ "${QWEN35_VLLM_OFFLINE_ENABLE_HMA}" == "1" ]]; then
    VLLM_OFFLINE_ARGS+=(--vllm-enable-hybrid-kv-cache-manager)
  fi
  if [[ -n "${QWEN35_VLLM_OFFLINE_CONNECTOR_MODULE_PATH}" ]]; then
    VLLM_OFFLINE_ARGS+=(--vllm-kv-connector-module-path "${QWEN35_VLLM_OFFLINE_CONNECTOR_MODULE_PATH}")
  fi

  run_python scripts/benchmark_future_imagination_latency.py \
    --provider vllm_offline \
    --model "${model_id}" \
    --vllm-model-id "${model_id}" \
    --vllm-tensor-parallel-size "${tp_size}" \
    --vllm-dtype "${QWEN35_VLLM_OFFLINE_DTYPE}" \
    --vllm-max-model-len "${MAX_MODEL_LEN}" \
    --vllm-gpu-memory-utilization "${gpu_util}" \
    --vllm-spec-num-tokens "${QWEN35_VLLM_OFFLINE_SPEC_NUM_TOKENS}" \
    --vllm-kv-connector-mode "${QWEN35_VLLM_OFFLINE_CONNECTOR_MODE}" \
    "${VLLM_OFFLINE_ARGS[@]}" \
    --prompt-dir-state "${PROMPT_STATE_DIR}" \
    --prompt-dir-history "${PROMPT_HISTORY_DIR}" \
    --benchmark-mode "${BENCHMARK_MODE}" \
    --warmup-prompts-per-variant "${WARMUP_PROMPTS_PER_VARIANT}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --repeats "${REPEATS}" \
    --timeout-s "${REQUEST_TIMEOUT_S}" \
    --hidden-target-layer -1 \
    --output-dir "${bench_out}" \
    --run-label "${label}"
}

if [[ -n "${GEMINI_API_KEY}" ]]; then
  echo "[step] run gemini benchmark (${GEMINI_MODEL})"
  run_python scripts/benchmark_future_imagination_latency.py \
    --provider gemini \
    --model "${GEMINI_MODEL}" \
    --api-key "${GEMINI_API_KEY}" \
    --prompt-dir-state "${PROMPT_STATE_DIR}" \
    --prompt-dir-history "${PROMPT_HISTORY_DIR}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --repeats "${REPEATS}" \
    --timeout-s "${REQUEST_TIMEOUT_S}" \
    --output-dir "${OUT_DIR}/gemini_flash" \
    --run-label "gemini_flash"
else
  echo "[warn] GEMINI_API_KEY not set; skipping Gemini benchmark."
fi

if [[ "${RUN_QWEN4B}" == "1" ]]; then
  run_qwen_bench "qwen3_4b_vllm_hidden" "${QWEN4B_CONFIG}" "${QWEN4B_CONFIG}" 1 "${QWEN4B_PORT}" "${QWEN4B_GPU_UTIL}"
else
  echo "[info] RUN_QWEN4B=0; skipping qwen3_4b_vllm_hidden"
fi

if [[ "${RUN_QWEN4B_HF}" == "1" ]]; then
  run_qwen_hf_bench "qwen3_4b_hf_hidden" "${QWEN4B_MODEL_ID}"
else
  echo "[info] RUN_QWEN4B_HF=0; skipping qwen3_4b_hf_hidden"
fi

if [[ "${RUN_QWEN8B}" == "1" ]]; then
  run_qwen_bench "qwen3_8b_vllm_hidden" "${QWEN8B_CONFIG_OUT}" "${QWEN8B_CONFIG_OUT}" "${QWEN8B_TP}" "${QWEN8B_PORT}" "${QWEN8B_GPU_UTIL}"
else
  echo "[info] RUN_QWEN8B=0; skipping qwen3_8b_vllm_hidden"
fi

if [[ "${RUN_QWEN8B_HF}" == "1" ]]; then
  run_qwen_hf_bench "qwen3_8b_hf_hidden" "${QWEN8B_MODEL_ID}"
else
  echo "[info] RUN_QWEN8B_HF=0; skipping qwen3_8b_hf_hidden"
fi

if [[ "${RUN_QWEN14B}" == "1" ]]; then
  run_qwen_bench "qwen3_14b_vllm_hidden" "${QWEN14B_CONFIG_OUT}" "${QWEN14B_CONFIG_OUT}" "${QWEN14B_TP}" "${QWEN14B_PORT}" "${QWEN14B_GPU_UTIL}"
else
  echo "[info] RUN_QWEN14B=0; skipping qwen3_14b_vllm_hidden"
fi

if [[ "${RUN_QWEN14B_HF}" == "1" ]]; then
  run_qwen_hf_bench "qwen3_14b_hf_hidden" "${QWEN14B_MODEL_ID}"
else
  echo "[info] RUN_QWEN14B_HF=0; skipping qwen3_14b_hf_hidden"
fi

if [[ "${RUN_QWEN9B}" == "1" ]]; then
  if [[ "${QWEN35_BACKEND}" == "vllm_hidden" ]]; then
    if [[ "${USE_QWEN35_COMPAT_MODEL}" == "1" ]]; then
      if [[ "${VLLM_RUNTIME_VERSION}" == 0.14.* ]]; then
        run_qwen_bench "qwen3_5_9b_vllm_hidden" "${QWEN9B_COMPAT_MODEL_DIR}" "${QWEN9B_COMPAT_MODEL_DIR}" "${QWEN9B_TP}" "${QWEN9B_PORT}" "${QWEN9B_GPU_UTIL}"
      else
        run_qwen_bench "qwen3_5_9b_vllm_hidden" "${QWEN9B_CONFIG_OUT}" "${QWEN9B_CONFIG_OUT}" "${QWEN9B_TP}" "${QWEN9B_PORT}" "${QWEN9B_GPU_UTIL}"
      fi
    else
      run_qwen_bench "qwen3_5_9b_vllm_hidden" "${QWEN9B_CONFIG_OUT}" "${QWEN9B_CONFIG_OUT}" "${QWEN9B_TP}" "${QWEN9B_PORT}" "${QWEN9B_GPU_UTIL}"
    fi
  elif [[ "${QWEN35_BACKEND}" == "vllm_offline" ]]; then
    QWEN35_9B_VLLM_OFFLINE_MODEL="${QWEN9B_MODEL_ID}"
    if [[ "${USE_QWEN35_COMPAT_MODEL}" == "1" ]]; then
      QWEN35_9B_VLLM_OFFLINE_MODEL="${QWEN9B_COMPAT_MODEL_DIR}"
    fi
    run_qwen_vllm_offline_bench "qwen3_5_9b_vllm_offline_hidden" "${QWEN35_9B_VLLM_OFFLINE_MODEL}" "${QWEN9B_TP}" "${QWEN9B_GPU_UTIL}"
  else
    echo "[info] qwen3.5-9b vLLM path disabled (QWEN35_BACKEND=${QWEN35_BACKEND})"
  fi
else
  echo "[info] RUN_QWEN9B=0; skipping qwen3.5-9b vLLM benchmark"
fi

if [[ "${RUN_QWEN9B_HF}" == "1" || ( "${RUN_QWEN9B}" == "1" && "${QWEN35_BACKEND}" == "hf_local" ) ]]; then
  run_qwen_hf_bench "qwen3_5_9b_hf_hidden" "${QWEN9B_MODEL_ID}"
else
  echo "[info] RUN_QWEN9B_HF=0; skipping qwen3_5_9b_hf_hidden"
fi

if [[ "${RUN_QWEN27B}" == "1" ]]; then
  if [[ "${QWEN35_BACKEND}" == "vllm_hidden" ]]; then
    if [[ "${USE_QWEN35_COMPAT_MODEL}" == "1" ]]; then
      if [[ "${VLLM_RUNTIME_VERSION}" == 0.14.* ]]; then
        run_qwen_bench "qwen3_5_27b_vllm_hidden" "${QWEN27B_COMPAT_MODEL_DIR}" "${QWEN27B_COMPAT_MODEL_DIR}" "${QWEN27B_TP}" "${QWEN27B_PORT}" "${QWEN27B_GPU_UTIL}"
      else
        run_qwen_bench "qwen3_5_27b_vllm_hidden" "${QWEN27B_CONFIG_OUT}" "${QWEN27B_CONFIG_OUT}" "${QWEN27B_TP}" "${QWEN27B_PORT}" "${QWEN27B_GPU_UTIL}"
      fi
    else
      run_qwen_bench "qwen3_5_27b_vllm_hidden" "${QWEN27B_CONFIG_OUT}" "${QWEN27B_CONFIG_OUT}" "${QWEN27B_TP}" "${QWEN27B_PORT}" "${QWEN27B_GPU_UTIL}"
    fi
  elif [[ "${QWEN35_BACKEND}" == "vllm_offline" ]]; then
    QWEN35_27B_VLLM_OFFLINE_MODEL="${QWEN27B_MODEL_ID}"
    if [[ "${USE_QWEN35_COMPAT_MODEL}" == "1" ]]; then
      QWEN35_27B_VLLM_OFFLINE_MODEL="${QWEN27B_COMPAT_MODEL_DIR}"
    fi
    run_qwen_vllm_offline_bench "qwen3_5_27b_vllm_offline_hidden" "${QWEN35_27B_VLLM_OFFLINE_MODEL}" "${QWEN27B_TP}" "${QWEN27B_GPU_UTIL}"
  else
    echo "[info] qwen3.5-27b vLLM path disabled (QWEN35_BACKEND=${QWEN35_BACKEND})"
  fi
else
  echo "[info] RUN_QWEN27B=0; skipping qwen3.5-27b vLLM benchmark"
fi

if [[ "${RUN_QWEN27B_HF}" == "1" || ( "${RUN_QWEN27B}" == "1" && "${QWEN35_BACKEND}" == "hf_local" ) ]]; then
  run_qwen_hf_bench "qwen3_5_27b_hf_hidden" "${QWEN27B_MODEL_ID}"
else
  echo "[info] RUN_QWEN27B_HF=0; skipping qwen3_5_27b_hf_hidden"
fi

echo "[step] build combined markdown"
OUT_DIR_FOR_PY="${OUT_DIR}" python3 - <<'PY'
import json
import os
import pathlib

out_dir = pathlib.Path(os.environ["OUT_DIR_FOR_PY"])
summary_rows = []

for child in sorted(out_dir.iterdir()):
    if not child.is_dir():
        continue
    summary_path = child / "summary.json"
    if not summary_path.exists():
        continue
    rows = json.loads(summary_path.read_text())
    for row in rows:
        row = dict(row)
        row["run"] = child.name
        summary_rows.append(row)

lines = []
lines.append("# Combined Latency Benchmark")
lines.append("")
lines.append(f"- output_dir: `{out_dir}`")
lines.append("")
lines.append("| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |")
lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in summary_rows:
    def fmt(x, nd=4):
        if x is None:
            return ""
        if isinstance(x, (int, float)):
            return f"{x:.{nd}f}"
        return str(x)
    lines.append(
        "| {run} | {model} | {variant} | {calls} | {ok_calls} | {error_calls} | {lat_mean} | {rps_mean} | {tot_lat_mean} | {tot_rps_mean} | {comp_mean} |".format(
            run=r.get("run", ""),
            model=r.get("model", ""),
            variant=r.get("variant", ""),
            calls=r.get("calls", ""),
            ok_calls=r.get("ok_calls", ""),
            error_calls=r.get("error_calls", ""),
            lat_mean=fmt(r.get("avg_request_latency_seconds")),
            rps_mean=fmt(r.get("avg_requests_per_second")),
            tot_lat_mean=fmt(r.get("avg_end_to_end_latency_seconds")),
            tot_rps_mean=fmt(r.get("avg_end_to_end_responses_per_second")),
            comp_mean=fmt(r.get("avg_completion_tokens"), 1),
        )
    )

(out_dir / "combined_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_dir / "combined_summary.md")
PY

echo "[done] out_dir=${OUT_DIR}"
