# vLLM Hidden States Extractor Config for Qwen3-4B

Config for `Qwen/Qwen3-4B-Thinking-2507` with the
[vllm-hidden-states-extractor](https://github.com/fynnsu/vllm-hidden-states-extractor) plugin.

## Quick Start

```bash
# On compute node with GPU:
conda activate craftax_fast_llm
cd ~/Craftax_Baselines

# Start server (patches connector + launches vLLM):
bash scripts/start_vllm_hidden.sh --mode last_token

# In another terminal, test:
python scripts/test_vllm_hidden.py
```

## Config Details

- **Model**: Qwen/Qwen3-4B-Thinking-2507 (Qwen3ForCausalLM)
- **Hidden size**: 2560, **head_dim**: 128, **36 layers**
- **Extracted layers**: 8, 16, 24, 35 (evenly distributed)
- **Speculator hidden_size**: 10240 (= 2560 × 4 layers)
- **num_attention_heads**: 20 (= 10240 / 128 / 4, accounting for internal 2x multiplier + k/v split)

## Connector Modes

Set via `kv_connector_extra_config.mode`:
- `"all"` — saves hidden states for all tokens `[4, seq_len, 2560]`
- `"last_token"` — saves only last token `[4, 1, 2560]` (for prompt-only extraction)
