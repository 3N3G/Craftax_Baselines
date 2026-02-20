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

set -e

# Parse arguments
MODE="all"
if [ "$1" = "--mode" ] && [ -n "$2" ]; then
    MODE="$2"
elif [ -n "$1" ] && [ "$1" != "--mode" ]; then
    MODE="$1"
fi

STORAGE_PATH="/tmp/hidden_states"
CONFIG_DIR="./configs/vllm_hidden_qwen4b"

# Find and patch the connector with our modified version
PLUGIN_DIR=$(python -c "import vllm_hidden_states_extractor; import os; print(os.path.dirname(vllm_hidden_states_extractor.__file__))")
echo "Plugin directory: $PLUGIN_DIR"

echo "Patching connector with last_token support..."
cp utils/vllm_hidden_connector.py "$PLUGIN_DIR/connector.py"
echo "✅ Connector patched"

# Clean old hidden states
rm -rf "$STORAGE_PATH"
mkdir -p "$STORAGE_PATH"

echo ""
echo "Starting vLLM server..."
echo "  Config: $CONFIG_DIR"
echo "  Mode: $MODE"
echo "  Storage: $STORAGE_PATH"
echo ""

vllm serve "$CONFIG_DIR" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config "{\"kv_connector\":\"ExampleHiddenStatesConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"$STORAGE_PATH\",\"mode\":\"$MODE\"}}"
