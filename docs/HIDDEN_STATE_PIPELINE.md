# Hidden State Pipeline Documentation

## Overview

The hidden state pipeline extracts LLM reasoning representations to train RL policies. All scripts use the **same standardized approach**:

1. Generate 256 tokens with `output_hidden_states=True`
2. Extract last layer hidden states for ALL 256 generated tokens
3. Mean pool across tokens → `(batch, 2560)` single vector per sample

## Scripts

### Training Data Generation

| Script | Purpose | Input | Output Shape |
|--------|---------|-------|--------------|
| `labelling/llm_worker.py` | Generate text + hidden states from unlabelled data | `obs` | `(batch, 256, 2560)` saved |
| `labelling/extract_hidden_states.py` | Re-extract hidden states from existing text | `text_generated` | `(batch, 256, 2560)` saved |

Both save the full 256 tokens. The training dataloader should mean-pool to `(batch, 2560)`.

### Online RL

| Script | Purpose | Output |
|--------|---------|--------|
| `online_rl_hidden.py` | Live hidden state extraction during training | `(batch, 2560)` mean-pooled |

Mean pools internally before passing to policy network.

### Evaluation Server

| Script | Purpose | Output |
|--------|---------|--------|
| `vlm_server.py` | Flask API for eval | `(2560,)` mean-pooled |

Used by `eval_awr_aug_client_proper.py` for evaluation.

## Architecture

The `ActorCriticConvAug` in `awr_aug.py` takes:
- `obs`: Image observation `(B, H, W, 3)`
- `hidden_state`: Mean-pooled LLM embedding `(B, 2560)`

Concatenates CNN features (512) + hidden state (2560) = 3072 features → Actor/Critic heads.

## Key Constants

```python
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"  # or Qwen3-VL-4B-Instruct for VLM
TOKENS_GENERATED = 256
HIDDEN_SIZE = 2560
```

## Data Format

NPZ files contain:
- `obs`: Observations
- `next_obs`: Next observations  
- `action`: Actions taken
- `reward`: Rewards
- `done`: Episode termination flags
- `log_prob`: Log probabilities (optional)
- `text_generated`: Generated text strings
- `hidden_state`: `(N, 256, 2560)` - all token hidden states

## Processing Flow

```
Observation → Prompt → LLM Generate (256 tokens) → Extract Last Layer Hidden States
                                                            ↓
                                                   (batch, 256, 2560)
                                                            ↓
                                                   Mean Pool (dim=1)
                                                            ↓
                                                   (batch, 2560) → Policy Network
```

## vLLM Server Configuration

### Layer Selection

The vLLM server uses an EAGLE-style configuration that extracts specific layers from the model. The default config (`configs/vllm_hidden_qwen4b/config.json`) extracts layers `[8, 16, 24, 35]`.

To extract from different layers:
1. Modify `eagle_aux_hidden_state_layer_ids` in the config
2. Restart the vLLM server

### Token Generation Modes

The online RL pipeline now supports two modes:

1. **Prompt-only mode** (`--tokens 1`):
   - No text generation, just forward pass through prompt
   - ~34x faster than generation mode
   - Deterministic (no sampling)
   - Better for observation discrimination

2. **Generation mode** (`--tokens N` where N > 1):
   - Generates N tokens before extracting hidden states
   - Allows for reasoning/CoT before extraction
   - Slower but may capture more semantic understanding

Usage:
```bash
# Prompt-only mode (fast, deterministic)
python online_rl_llm/online_rl_hidden.py --envs 128 --steps 100000 --skip-n 25 --layer -1 --tokens 1

# Generation mode with 64 tokens
python online_rl_llm/online_rl_hidden.py --envs 128 --steps 100000 --skip-n 25 --layer -1 --tokens 64

# Use layer 24 instead of last layer
python online_rl_llm/online_rl_hidden.py --envs 128 --steps 100000 --skip-n 25 --layer 24 --tokens 1
```

## Changelog

- **2026-02-12**: Added `--layer` and `--tokens` flags to online RL for flexible hidden state extraction
- **2026-02-07**: Standardized all scripts to use all 256 tokens with mean pooling (removed subsampling from vlm_server.py)
