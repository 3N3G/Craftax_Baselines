#!/bin/bash
#SBATCH --job-name=latbench_qwen_fast
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=logs/latbench_qwen_fast_%j.out
#SBATCH --error=logs/latbench_qwen_fast_%j.err

set -eo pipefail
if [[ "${DEBUG_LATBENCH_SBATCH:-0}" == "1" ]]; then
  set -x
fi

cd ~/Craftax_Baselines

USER_NAME="${USER:-${LOGNAME:-geney}}"
SRC_ENV_PATH="${SRC_ENV_PATH:-/data/user_data/${USER_NAME}/.conda/envs/craftax_fast_llm}"
TARGET_VLLM_VERSION="${TARGET_VLLM_VERSION:-0.17.1}"
TRANSFORMERS_GIT_REF="${TRANSFORMERS_GIT_REF:-main}"

if [[ -z "${OVERLAY_ROOT:-}" ]]; then
  OVERLAY_ROOT="/scratch/${USER_NAME}/latbench_overlay_${SLURM_JOB_ID}"
fi
if ! mkdir -p "${OVERLAY_ROOT}" 2>/dev/null; then
  OVERLAY_ROOT="${SLURM_TMPDIR:-/tmp}/latbench_overlay_${SLURM_JOB_ID}"
  mkdir -p "${OVERLAY_ROOT}"
fi
OVERLAY_ENV_PATH="${OVERLAY_ENV_PATH:-${OVERLAY_ROOT}/venv}"
HF_OVERLAY_ENV_PATH="${HF_OVERLAY_ENV_PATH:-${OVERLAY_ROOT}/hf_venv}"
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
echo "[info] hf_overlay_env=${HF_OVERLAY_ENV_PATH}"
echo "[info] target_vllm_version=${TARGET_VLLM_VERSION}"
echo "[info] tmpdir=${TMPDIR}"
echo "[info] xdg_cache_home=${XDG_CACHE_HOME}"
echo "[info] pip_cache_dir=${PIP_CACHE_DIR}"
echo "[info] pip_no_cache_dir=${PIP_NO_CACHE_DIR}"

echo "[step] create clean overlay venv"
"${SRC_ENV_PATH}/bin/python" -m venv --system-site-packages "${OVERLAY_ENV_PATH}"

echo "[step] upgrade pip tooling in overlay env"
PIP_DISABLE_PIP_VERSION_CHECK=1 "${OVERLAY_ENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel

echo "[step] install benchmark deps in overlay env"
PIP_DISABLE_PIP_VERSION_CHECK=1 "${OVERLAY_ENV_PATH}/bin/python" -m pip install \
  "vllm==${TARGET_VLLM_VERSION}" \
  "accelerate>=0.33.0" \
  "safetensors>=0.4.0"

echo "[step] overlay probe"
"${OVERLAY_ENV_PATH}/bin/python" - <<PY
import os
import sys
print("python_exe", sys.executable)
import vllm
print("vllm_version", vllm.__version__)
print("vllm_path", os.path.dirname(vllm.__file__))
import transformers
print("transformers_version", transformers.__version__)
PY

echo "[step] create hf overlay venv"
"${SRC_ENV_PATH}/bin/python" -m venv --system-site-packages "${HF_OVERLAY_ENV_PATH}"

echo "[step] upgrade pip tooling in hf overlay env"
PIP_DISABLE_PIP_VERSION_CHECK=1 "${HF_OVERLAY_ENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel

echo "[step] install source transformers in hf overlay env"
PIP_DISABLE_PIP_VERSION_CHECK=1 "${HF_OVERLAY_ENV_PATH}/bin/python" -m pip install --upgrade \
  "git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_GIT_REF}"

echo "[step] hf overlay probe"
"${HF_OVERLAY_ENV_PATH}/bin/python" - <<PY
import os
import sys
print("hf_python_exe", sys.executable)
import transformers
print("hf_transformers_version", transformers.__version__)
import huggingface_hub
print("hf_hub_version", huggingface_hub.__version__)
PY

PATCH_VLLM_CONNECTOR="${PATCH_VLLM_CONNECTOR:-1}"
if [[ "${PATCH_VLLM_CONNECTOR}" == "1" ]]; then
  echo "[step] patch vLLM get_hf_text_config for dict text_config compatibility"
  "${OVERLAY_ENV_PATH}/bin/python" - <<'PY'
import inspect
import pathlib

import vllm.transformers_utils.config as cfg_mod

path = pathlib.Path(inspect.getfile(cfg_mod))
text = path.read_text(encoding="utf-8")
orig = text
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
    "        # Accept nested dict text_config objects used by qwen3.5 configs.\n"
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
if needle in text:
    text = text.replace(needle, replace, 1)
if text != orig:
    path.write_text(text, encoding="utf-8")
print(f"patched_hf_text_config={path}")
PY

  echo "[step] patch vLLM ExampleHiddenStatesConnector for HMA + empty block metadata"
  "${OVERLAY_ENV_PATH}/bin/python" - <<'PY'
import inspect
import pathlib

import vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector as connector

path = pathlib.Path(inspect.getfile(connector))
text = path.read_text(encoding="utf-8")
orig = text

if "SupportsHMA" not in text:
    text = text.replace(
        "from vllm.distributed.kv_transfer.kv_connector.v1.base import (\n"
        "    KVConnectorBase_V1,\n"
        "    KVConnectorMetadata,\n"
        "    KVConnectorRole,\n"
        ")\n",
        "from vllm.distributed.kv_transfer.kv_connector.v1.base import (\n"
        "    KVConnectorBase_V1,\n"
        "    KVConnectorMetadata,\n"
        "    KVConnectorRole,\n"
        "    SupportsHMA,\n"
        ")\n",
    )
    text = text.replace(
        "class ExampleHiddenStatesConnector(KVConnectorBase_V1):",
        "class ExampleHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):",
    )
    needle = (
        "    def request_finished(\n"
        "        self,\n"
        "        request: \"Request\",\n"
        "        block_ids: list[int],\n"
        "    ) -> tuple[bool, dict[str, Any] | None]:\n"
    )
    insert = (
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
    if needle in text:
        text = text.replace(needle, insert)

repls = [
    (
        "block_ids=new_req.block_ids[0],",
        "block_ids=(new_req.block_ids[0] if new_req.block_ids else []),",
    ),
    (
        "self._req_blocks[new_req.req_id] = list(new_req.block_ids[0])",
        "self._req_blocks[new_req.req_id] = list(new_req.block_ids[0]) if new_req.block_ids else []",
    ),
    (
        "            assert new_block_ids is not None\n            block_ids = new_block_ids[0]",
        "            if not new_block_ids:\n                block_ids = []\n            else:\n                block_ids = new_block_ids[0] or []",
    ),
    (
        "            req_block_ids.extend(block_ids)\n            filename = os.path.join(self._storage_path, f\"{req_id}.safetensors\")",
        "            if not block_ids and not req_block_ids:\n                continue\n\n            req_block_ids.extend(block_ids)\n            filename = os.path.join(self._storage_path, f\"{req_id}.safetensors\")",
    ),
]

applied = 0
for src, dst in repls:
    if src in text:
        text = text.replace(src, dst)
        applied += 1

if text != orig:
    path.write_text(text, encoding="utf-8")
print(f"patched_connector_file={path}")
print(f"patch_replacements_applied={applied}")
PY

  echo "[step] patch vLLM gpu_model_runner for qwen3.5 aux-hidden extra outputs"
  "${OVERLAY_ENV_PATH}/bin/python" - <<'PY'
import inspect
import pathlib

import vllm.v1.worker.gpu_model_runner as gpu_model_runner

path = pathlib.Path(inspect.getfile(gpu_model_runner))
text = path.read_text(encoding="utf-8")
orig = text

src_post = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                # True when EAGLE 3 is used.\n"
    "                hidden_states, aux_hidden_states = model_output\n"
    "            else:\n"
)
dst_post = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                # True when EAGLE 3 is used. Some model forwards return\n"
    "                # extra bookkeeping outputs; keep the first two.\n"
    "                if not isinstance(model_output, (tuple, list)):\n"
    "                    raise RuntimeError(\n"
    "                        \"Expected tuple/list model_output when aux hidden states are enabled\"\n"
    "                    )\n"
    "                if len(model_output) < 2:\n"
    "                    raise RuntimeError(\n"
    "                        f\"Expected at least 2 outputs for aux hidden states, got {len(model_output)}\"\n"
    "                    )\n"
    "                hidden_states = model_output[0]\n"
    "                aux_hidden_states = model_output[1]\n"
    "            else:\n"
)
if src_post in text:
    text = text.replace(src_post, dst_post, 1)

src_dummy = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                hidden_states, _ = outputs\n"
    "            else:\n"
)
dst_dummy = (
    "            if self.use_aux_hidden_state_outputs:\n"
    "                if isinstance(outputs, (tuple, list)):\n"
    "                    if len(outputs) < 1:\n"
    "                        raise RuntimeError(\"Expected at least one output tensor from model forward\")\n"
    "                    hidden_states = outputs[0]\n"
    "                else:\n"
    "                    hidden_states = outputs\n"
    "            else:\n"
)
if src_dummy in text:
    text = text.replace(src_dummy, dst_dummy, 1)

if text != orig:
    path.write_text(text, encoding="utf-8")
print(f"patched_gpu_model_runner={path}")
PY
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

RUN_TAG="${RUN_TAG:-qwen_native_vs_hf_fastpath}" \
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
QWEN27B_TP="${QWEN27B_TP:-2}" \
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}" \
SAFE_MODE="${SAFE_MODE:-0}" \
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}" \
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}" \
VLLM_PYTHON_BIN="${OVERLAY_ENV_PATH}/bin/python" \
PREP_PYTHON_BIN="${OVERLAY_ENV_PATH}/bin/python" \
HF_PYTHON_BIN="${HF_OVERLAY_ENV_PATH}/bin/python" \
env -u BASH_ENV bash --noprofile --norc scripts/shell/run_future_latency_qwen3_native_vs_hf.sh

echo "[done] qwen native-vllm vs hf fast-path benchmark finished"
