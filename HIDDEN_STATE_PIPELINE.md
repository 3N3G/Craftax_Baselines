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

## Changelog

- **2026-02-07**: Standardized all scripts to use all 256 tokens with mean pooling (removed subsampling from vlm_server.py)
