# vLLM Hidden States Extractor Config for Qwen3-4B

This directory contains the config to use `Qwen/Qwen3-4B-Thinking-2507` with the 
[vllm-hidden-states-extractor](https://github.com/fynnsu/vllm-hidden-states-extractor) plugin.

## Setup

```bash
# Install the plugin
pip install git+https://github.com/fynnsu/vllm-hidden-states-extractor.git
pip install safetensors
```

## Usage

Start the vLLM server with hidden states extraction:

```bash
vllm serve ./vllm_hidden_qwen4b \
    --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/tmp/hidden_states"}}'
```

## Config Details

- **Model**: Qwen/Qwen3-4B-Thinking-2507
- **Hidden layers**: 36 total
- **Extracted layers**: 8, 16, 24, 35 (distributed across depth)
- **Hidden size**: 2560

The `eagle_aux_hidden_state_layer_ids` specifies which layers to extract hidden states from.
The `speculator.speculator_type: "extract_hidden_states"` tells vLLM to use the plugin's dummy model.
