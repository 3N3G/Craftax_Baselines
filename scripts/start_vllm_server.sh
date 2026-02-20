#!/bin/bash
# Start vLLM server with hidden states extraction

# Use single-layer config if available, otherwise use 4-layer
CONFIG_DIR="configs/vllm_hidden_last"
if [ ! -f "$CONFIG_DIR/config.json" ]; then
    CONFIG_DIR="configs/vllm_hidden_qwen4b"
    echo "Using 4-layer config: $CONFIG_DIR"
else
    echo "Using single-layer config: $CONFIG_DIR"
fi

echo "Starting vLLM server..."
echo "Config: $CONFIG_DIR"

# Start server (all on one line)
vllm serve "$CONFIG_DIR" --max-model-len 8192 --gpu-memory-utilization 0.95 --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/tmp/hidden_states","mode":"last_token"}}'