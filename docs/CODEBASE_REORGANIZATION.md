# Craftax_Baselines Codebase Summary & Reorganization Proposal

## Current Codebase Overview

This codebase implements RL experiments for the Craftax game, combining traditional RL with LLM-based approaches. The current structure is **flat and disorganized**, with 70+ files in the root directory making navigation difficult.

---

## Detailed File-by-File Summary

### 1. Online RL Training (JAX/Flax - Data Generation)

| File | Purpose | Framework | Notes |
|------|---------|-----------|-------|
| `ppo.py` | Core PPO with ICM intrinsic motivation (optional) | JAX/Flax | Saves trajectories for offline training; 905 lines |
| `ppo_rnd.py` | PPO with Random Network Distillation | JAX/Flax | Alternative intrinsic reward |
| `ppo_rnn.py` | PPO with recurrent policy | JAX/Flax | GRU-based memory |
| `ppo_pixel.py` | PPO for pixel observations with Gemma embeddings | JAX/Flax | Experimental, uses `gemma` package |
| `run_ppo_floor_logging.py` | PPO variant that logs floor progression | JAX/Flax | Debugging/analysis tool |

**SBATCH Scripts:**
- `run_ppo_symbolic.sbatch`, `run_ppo_video.sbatch` - PPO job submission
- `run_ppo_floor_logging.sbatch` - Floor logging variant

---

### 2. LLM Direct Play (Testing Prompts)

| File | Purpose | Notes |
|------|---------|-------|
| `llm_play.py` | Basic LLM agent using Qwen to play Craftax | 1313 lines; contains SYSTEM_PROMPT |
| `llm_play_harnessed.py` | Enhanced version with history context & filtered obs | 920 lines; production version |
| `query_qwen.py` | Interactive REPL for testing LLM prompts | CLI tool for prompt engineering |

---

### 3. Online RL + LLM Hidden State Extraction

| File | Purpose | Notes |
|------|---------|-------|
| `online_rl_hidden.py` | Online PPO with live LLM hidden state extraction | 877 lines; HuggingFace + Flash Attention |
| `online_rl_smoke_test.py` | Smoke test with SGLang server | Validates setup |
| `online_rl_vllm_test.py` | Smoke test with vLLM | Validates setup |
| `vllm_policy.py` | vLLM policy wrapper for batch inference | 424 lines |
| `sglang_policy.py` | SGLang policy wrapper with async requests | 276 lines |

**SBATCH Scripts:**
- `run_online_rl_hidden.sbatch`, `run_online_rl_test.sbatch`, `run_vllm_rl_test.sbatch`

---

### 4. Offline RL Training (PyTorch)

| File | Purpose | Notes |
|------|---------|-------|
| `pt_awr.py` | Advantage Weighted Regression (baseline, no LLM) | 656 lines; PyTorch |
| `pt_bc.py` | Behavior Cloning (baseline, no LLM) | 445 lines; PyTorch |
| `awr_aug.py` | AWR augmented with LLM hidden states | 449 lines; uses `hidden_state` input |
| `awr_aug_onlygeneration.py` | **DUPLICATE** of `awr_aug.py` | Nearly identical; delete candidate |
| `ppo_add_returns.py` | Post-processing script to add returns to trajectories | Python | Utility for offline data |


**Evaluation Scripts:**
- `pt_eval_awr.py` - Evaluate `pt_awr.py` checkpoints
- `eval_awr_aug_client_proper.py` - Evaluate `awr_aug.py` using VLM server for hidden states

**Shell Scripts:**
- `awr_baseline.sh`, `run_augmented_awr.sh`, `run_train.sh`

---

### 5. Labelling Pipeline (Hidden State Extraction)

Located in `labelling/` (cleaned up — only 8 files):

| File | Purpose | Notes |
|------|---------|-------|
| `llm_worker.py` | vLLM-based worker for hidden state extraction | Queue-based; saves (N, hidden_size) |
| `obs_to_text.py` | Convert symbolic observations to text | Used by llm_worker |
| `add_text_obs.py` | Add text observations to existing NPZ files | Utility |
| `janitor_llm.py` | Re-queue failed/orphaned jobs | Queue maintenance |
| `addtoqueue_llm.py` | Add files to Redis queue | Queue management |

**SBATCH Scripts:**
- `makeworkers_llm.sbatch` - Spawn GPU workers (each starts own vLLM server)
- `run_labelling.sbatch` - End-to-end coordinator (Redis + queue + workers + monitor)
- `run_extract_hidden.sbatch` - Re-extract hidden states for existing data

---

### 6. Benchmarking

| File | Purpose | Notes |
|------|---------|-------|
| `benchmark_inference.py` | Compare HuggingFace vs vLLM vs SGLang | 354 lines |
| `benchmark_hidden_states.py` | Benchmark hidden state extraction methods | 619 lines |
| `benchmark_vllm_hidden_plugin.py` | Benchmark vLLM hidden states plugin | 504 lines |

**SBATCH Scripts:**
- `run_benchmark.sbatch`, `run_hidden_bench.sbatch`, `run_sglang_benchmark.sbatch`

---

### 7. VLM Server (Inference Service)

| File | Purpose | Notes |
|------|---------|-------|
| `vlm_server.py` | Flask API for hidden state extraction | Used by `eval_awr_aug_client_proper.py` |
| `submit_vlm_server.sh` | Submit VLM server job | |

---

### 8. Utilities & Shared

| File | Purpose | Notes |
|------|---------|-------|
| `wrappers.py` | JAX environment wrappers (BatchEnv, AutoReset, LogWrapper) | Core infrastructure |
| `image_utils.py` | Image format conversion utilities | Used by evaluation |
| `render_text_obs.py` | Render text observations as images | Visualization |

---

### 9. Models (Neural Networks)

Located in `models/`:

| File | Purpose | Framework |
|------|---------|-----------|
| `actor_critic.py` | Actor-Critic architectures (Conv, Symbolic, ImAug) | Flax/JAX |
| `rnd.py` | Random Network Distillation | Flax/JAX |
| `icm.py` | Intrinsic Curiosity Module | Flax/JAX |

---

### 10. Human Play & Recording

| File | Purpose | Notes |
|------|---------|-------|
| `play_craftax_recorder.py` | Pygame interface for human play with recording | Creates few-shot examples |

**Golden Examples:**
- `golden_examples/` - Recorded human gameplay sessions (JSONL + images)

---

### 11. Analysis & Debugging

| File | Purpose | Notes |
|------|---------|-------|
| `analysis/view_ppo_agent.py` | Load and visualize trained PPO agents | |
| `collectdatasetstats.py` | Compute dataset statistics | |
| `compute_hidden_stats.py` | Compute hidden state statistics | |
| `recompute_hidden_stats_from_server.py` | Recompute hidden stats via server | |
| `diagnose_hidden_state_mismatch.py` | Debug hidden state issues | |

---

### 12. Documentation

| File | Purpose |
|------|---------|
| `CLAUDE.md` | LLM prompt documentation and game strategy |
| `oldCLAUDE.md` | Older version of prompt documentation |
| `example-CLAUDE.md` | Example CLAUDE.md template |
| `docs/HIDDEN_STATE_PIPELINE.md` | Hidden state pipeline documentation |
| `docs/llm_labelling.md` | LLM labelling documentation |
| `docs/symbolic_to_pixels.md` | Symbolic to pixels conversion docs |
| `docs/README.md` | Docs index |

---

### 13. Example/Text Files

| File | Purpose | Action |
|------|---------|--------|
| `example_step.txt` | Example step output | **DELETE** - obsolete |
| `example_table_issue.txt` | Debug example | **DELETE** - obsolete |
| `llm_response_examples.txt` | Example LLM responses | **DELETE** - move to docs if needed |
| `llm_failed_extraction.txt` | Failed extraction logs | **DELETE** - debug artifact |
| `geminideepresearchllminference.txt` | Research notes | **DELETE** - move to docs if needed |
| `q1.txt` | Unknown | **DELETE** - unclear purpose |

---

### 14. Infrastructure Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Docker container definition |
| `requirements.txt` | Python dependencies |
| `.pre-commit-config.yaml` | Pre-commit hooks |
| `.gitignore` | Git ignore patterns |
| `build.sh`, `run_docker.sh`, `create_imaug_env.sh`, `fix_imaug.sh` | Setup scripts |

---

### 15. vLLM Plugin Config

Located in `vllm_hidden_qwen4b/`:
- `config.json` - Custom Qwen model config for hidden state extraction
- `README.md` - Plugin documentation

---

## Proposed File Deletions

| File | Reason |
|------|--------|
| `awr_aug_onlygeneration.py` | **Duplicate** of `awr_aug.py` (diff shows only minor config changes) |
| `example_step.txt` | Debug/example file, obsolete |
| `example_table_issue.txt` | Debug file, obsolete |
| `llm_failed_extraction.txt` | Debug artifact |
| `q1.txt` | Unclear purpose |
| `oldCLAUDE.md` | Superseded by `CLAUDE.md` | Check to see if any useful info before deletion |
| `example-CLAUDE.md` | Template, not needed |

KEEP BUT move if necessary to notebooks or something
| `labelling/trystuff.ipynb` | 217KB notebook - move to separate notebooks folder or delete |
---

## Proposed Directory Structure

```
Craftax_Baselines/
├── README.md
├── CLAUDE.md                      # Keep at root for visibility
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .pre-commit-config.yaml
│
├── configs/                       # NEW: All configuration files
│   └── vllm_hidden_qwen4b/        # Move from root
│       ├── config.json
│       └── README.md
│
├── docs/                          # EXISTING: Keep and expand
│   ├── README.md
│   ├── HIDDEN_STATE_PIPELINE.md
│   ├── llm_labelling.md
│   ├── symbolic_to_pixels.md
│   └── research_notes.md          # Optional: consolidate txt files
│
├── models/                        # EXISTING: Keep as-is
│   ├── __init__.py
│   ├── actor_critic.py
│   ├── icm.py
│   └── rnd.py
│
├── online_rl/                     # NEW: All online JAX RL
│   ├── __init__.py
│   ├── ppo.py
│   ├── ppo_rnd.py
│   ├── ppo_rnn.py
│   ├── ppo_pixel.py
│   └── ppo_floor_logging.py       # Rename from run_ppo_floor_logging.py
│
├── offline_rl/                    # NEW: All offline PyTorch RL
│   ├── __init__.py
│   ├── awr.py                     # Rename from pt_awr.py
│   ├── awr_augmented.py           # Rename from awr_aug.py
│   ├── bc.py                      # Rename from pt_bc.py
│   └── eval_awr.py                # Rename from pt_eval_awr.py
│   └── eval_awr_augmented.py      # Rename from eval_awr_aug_client_proper.py
│
├── online_rl_llm/                 # NEW: Online RL with LLM augmentation
│   ├── __init__.py
│   ├── online_rl_hidden.py
│   ├── vllm_policy.py
│   ├── sglang_policy.py
│   └── smoke_test_sglang.py       # Rename from online_rl_smoke_test.py
│   └── smoke_test_vllm.py         # Rename from online_rl_vllm_test.py
│
├── llm_play/                      # NEW: LLM direct play testing
│   ├── __init__.py
│   ├── llm_play.py
│   ├── llm_play_harnessed.py
│   └── query_qwen.py
│
├── labelling/                     # Cleaned up (8 files)
│   ├── llm_worker.py              # Main vLLM worker
│   ├── obs_to_text.py             # Symbolic obs → text
│   ├── add_text_obs.py            # Add text_obs to NPZ
│   ├── janitor_llm.py             # Re-queue failed jobs
│   ├── addtoqueue_llm.py          # Queue files
│   ├── makeworkers_llm.sbatch     # Spawn workers
│   ├── run_labelling.sbatch       # End-to-end coordinator
│   └── run_extract_hidden.sbatch  # Re-extract hidden states
│
├── servers/                       # NEW: Inference servers
│   ├── __init__.py
│   └── vlm_server.py
│
├── benchmarks/                    # NEW: All benchmarking code
│   ├── __init__.py
│   ├── benchmark_inference.py
│   ├── benchmark_hidden_states.py
│   └── benchmark_vllm_hidden_plugin.py
│
├── tools/                         # NEW: Human play, visualization, analysis
│   ├── __init__.py
│   ├── play_craftax_recorder.py
│   ├── render_text_obs.py
│   ├── view_ppo_agent.py          # Move from analysis/
│   ├── collectdatasetstats.py
│   ├── compute_hidden_stats.py
│   ├── diagnose_hidden_state_mismatch.py
│   └── ppo_add_returns.py
│
├── utils/                         # NEW: Shared utilities
│   ├── __init__.py
│   ├── wrappers.py
│   └── image_utils.py
│
├── golden_examples/               # EXISTING: Keep as-is
│   └── game_<timestamp>/
│
├── logz/                          # EXISTING: Keep as-is
│   └── batch_logging.py
│
├── scripts/                       # NEW: All shell and sbatch scripts
│   ├── shell/
│   │   ├── build.sh
│   │   ├── run_docker.sh
│   │   ├── create_imaug_env.sh
│   │   ├── fix_imaug.sh
│   │   ├── awr_baseline.sh
│   │   ├── run_augmented_awr.sh
│   │   ├── run_train.sh
│   │   ├── submit_vlm_server.sh
│   │   ├── submit_aug_eval_array.sh
│   │   ├── eval_batch_augmented.sh
│   │   └── eval_batch_baseline.sh
│   │
│   └── sbatch/
│       ├── run_ppo_symbolic.sbatch
│       ├── run_ppo_video.sbatch
│       ├── run_ppo_floor_logging.sbatch
│       ├── run_online_rl_hidden.sbatch
│       ├── run_online_rl_test.sbatch
│       ├── run_vllm_rl_test.sbatch
│       ├── run_benchmark.sbatch
│       ├── run_hidden_bench.sbatch
│       ├── run_sglang_benchmark.sbatch
│       ├── test_gpu.sbatch
│       ├── test_text_render.sbatch
│       ├── makeworkers_llm.sbatch
│       ├── run_labelling.sbatch
│       └── run_extract_hidden.sbatch
│
└── analysis/                      # Keep but maybe merge into tools/
    └── __init__.py
```

---

## Import Path Updates Required

When reorganizing, these imports need updating:

1. **`models/actor_critic.py`** is imported by many files:
   - `ppo.py`, `ppo_rnd.py`, `ppo_rnn.py` → `from models.actor_critic import ...`
   - Need to update to something like `from craftax_baselines.models.actor_critic import ...`

2. **`wrappers.py`** is imported by:
   - `ppo.py`, `ppo_rnd.py`, `ppo_rnn.py`, `ppo_pixel.py`

3. **`labelling/obs_to_text.py`** is imported by:
   - `run_ppo_floor_logging.py`

4. **`image_utils.py`** is imported by:
   - `pt_eval_awr.py`, `eval_awr_aug_client_proper.py`

5. **`awr_aug.py`** is imported by:
   - `eval_awr_aug_client_proper.py`

6. **`pt_awr.py`** is imported by:
   - `pt_eval_awr.py`

---

## Recommended Execution Order

1. **Delete obvious candidates first:**
   - `example_step.txt`, `example_table_issue.txt`, `llm_failed_extraction.txt`, `q1.txt`
   - `oldCLAUDE.md`, `example-CLAUDE.md`
   - Verify `awr_aug_onlygeneration.py` is duplicate, then delete

2. **Create new directories:**
   - `online_rl/`, `offline_rl/`, `online_rl_llm/`, `llm_play/`, `servers/`, `benchmarks/`, `tools/`, `utils/`, `scripts/shell/`, `scripts/sbatch/`, `configs/`

3. **Move Python files and update imports:**
   - Move files to new locations
   - Update import statements in each file
   - Add `__init__.py` files

4. **Move shell/sbatch scripts**

5. **Test that everything still works**

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Python files to organize | ~50 |
| Files to delete | ~8 |
| New directories to create | ~10 |
| Import paths to update | ~20-30 |

**Estimated effort:** 2-3 hours for complete reorganization with import fixes.
