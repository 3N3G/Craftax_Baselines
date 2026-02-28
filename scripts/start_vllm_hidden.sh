#!/bin/bash
# Start vLLM server with Hidden States Extractor Plugin
#
# Usage:
#   bash scripts/start_vllm_hidden.sh                    # all tokens mode
#   bash scripts/start_vllm_hidden.sh --mode last_token  # last token only
#
# Prerequisites:
#   conda activate craftax_fast_llm
#   pip install vllm-hidden-states-extractor  (already installed)

set -euo pipefail

# Parse arguments
MODE="all"
SAFE_MODE=0
PORT="${VLLM_PORT:-8000}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:-all}"
            shift 2
            ;;
        --port)
            PORT="${2:-$PORT}"
            shift 2
            ;;
        --safe)
            SAFE_MODE=1
            shift
            ;;
        *)
            MODE="$1"
            shift
            ;;
    esac
done

TMP_ROOT="${VLLM_TMP_ROOT:-/tmp/${USER:-$(id -un)}/craftax_vllm}"
STORAGE_PATH="${TMP_ROOT}/hidden_states"
CONFIG_SRC="./configs/vllm_hidden_qwen4b"
CONFIG_DIR="${TMP_ROOT}/vllm_hidden_qwen4b"
GPU_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-./configs/vllm_hidden_qwen4b}"

mkdir -p \
    "${TMP_ROOT}" \
    "${STORAGE_PATH}" \
    "${TMP_ROOT}/tmp" \
    "${TMP_ROOT}/torchinductor" \
    "${TMP_ROOT}/.cache" \
    "${TMP_ROOT}/pycache" \
    "${TMP_ROOT}/hf_home/hub" \
    "${TMP_ROOT}/hf_home/transformers" \
    "${TMP_ROOT}/vllm_cache"

# Keep vLLM/torch compile artifacts off quota-limited home paths.
export TMPDIR="${TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export TORCHINDUCTOR_CACHE_DIR="${TMP_ROOT}/torchinductor"
export XDG_CACHE_HOME="${TMP_ROOT}/.cache"
export PYTHONPYCACHEPREFIX="${TMP_ROOT}/pycache"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${TMP_ROOT}/hf_home"
export HUGGINGFACE_HUB_CACHE="${TMP_ROOT}/hf_home/hub"
export TRANSFORMERS_CACHE="${TMP_ROOT}/hf_home/transformers"
export VLLM_CACHE_ROOT="${TMP_ROOT}/vllm_cache"

if [[ ! -d "${CONFIG_SRC}" ]]; then
    echo "ERROR: missing vLLM config directory: ${CONFIG_SRC}"
    exit 1
fi

# Clear stale files to avoid corrupt cache reuse and stale file handle issues.
rm -rf "${TORCHINDUCTOR_CACHE_DIR:?}/"* || true
rm -rf "${STORAGE_PATH:?}/"* || true
rm -rf "${CONFIG_DIR:?}/"* || true

# Stage config onto local scratch to avoid metadata storms on shared storage.
if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${CONFIG_SRC}/" "${CONFIG_DIR}/"
else
    mkdir -p "${CONFIG_DIR}"
    cp -a "${CONFIG_SRC}/." "${CONFIG_DIR}/"
fi

# Find and patch the connector with our modified version
PLUGIN_DIR=$(python -c "import vllm_hidden_states_extractor; import os; print(os.path.dirname(vllm_hidden_states_extractor.__file__))")
echo "Plugin directory: $PLUGIN_DIR"

echo "Patching connector with last_token support..."
cp utils/vllm_hidden_connector.py "$PLUGIN_DIR/connector.py"
echo "✅ Connector patched"

echo ""
echo "Starting vLLM server..."
echo "  Config: $CONFIG_DIR"
echo "  Mode: $MODE"
echo "  Storage: $STORAGE_PATH"
echo "  Port: $PORT"
echo "  GPU memory utilization: $GPU_UTILIZATION"
echo "  Max model len: $MAX_MODEL_LEN"
echo "  Served model name: $SERVED_MODEL_NAME"
echo "  Safe mode: $SAFE_MODE"
echo ""

EXTRA_ARGS=()
if [[ "$SAFE_MODE" -eq 1 ]]; then
    # Avoid torch compile/cudagraph cache instability; prioritize reliability.
    EXTRA_ARGS+=(--enforce-eager)
fi

vllm serve "$CONFIG_DIR" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --gpu-memory-utilization "$GPU_UTILIZATION" \
    --kv-transfer-config "{\"kv_connector\":\"ExampleHiddenStatesConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"$STORAGE_PATH\",\"mode\":\"$MODE\"}}" \
    "${EXTRA_ARGS[@]}"
