# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

**Craftax Baselines** is a research repository for training RL agents on the Craftax environment—a procedurally-generated Minecraft-inspired gridworld. The codebase combines:
- **Online RL** (JAX/Flax) - PPO variants for data generation
- **Offline RL** (PyTorch) - AWR/BC from collected trajectories  
- **LLM-augmented RL** - Using VLM hidden states to improve policies
- **Distributed infrastructure** - Redis-based job queues for labelling

**Paper**: https://arxiv.org/abs/2402.16801  
**Craftax Repository**: https://github.com/MichaelTMatthews/Craftax/

---

## Directory Structure

```
Craftax_Baselines/
├── online_rl/          # JAX PPO variants (ppo.py, ppo_rnd.py, ppo_rnn.py, ppo_pixel.py)
├── offline_rl/         # PyTorch AWR/BC (awr.py, bc.py, awr_augmented.py, eval_*.py)
├── online_rl_llm/      # LLM-augmented online RL (online_rl_hidden.py, vllm_policy.py)
├── llm_play/           # LLM direct play testing (llm_play.py, llm_play_harnessed.py)
├── labelling/          # Distributed labelling pipeline (workers, queues, extractors)
├── servers/            # VLM inference server (vlm_server.py)
├── benchmarks/         # Performance benchmarking scripts
├── tools/              # Utilities (play_craftax_recorder.py, compute_hidden_stats.py)
├── utils/              # Shared code (wrappers.py, image_utils.py)
├── models/             # Neural network architectures (actor_critic.py, icm.py, rnd.py)
├── logz/               # Logging utilities
├── scripts/            # Shell and SLURM scripts
│   ├── shell/          # Shell scripts (awr_baseline.sh, run_augmented_awr.sh)
│   └── sbatch/         # SLURM batch scripts (run_ppo_symbolic.sbatch, etc.)
├── configs/            # Configuration files (vllm_hidden_qwen4b/)
├── golden_examples/    # Recorded human gameplay for few-shot prompting
└── docs/               # Documentation
    ├── llm_craftax_prompting.md  # LLM player prompt documentation
    ├── DEVELOPER_GUIDE.md        # Development practices & detailed reference
    ├── HIDDEN_STATE_PIPELINE.md  # Hidden state extraction docs
    ├── COMMANDS_REFERENCE.md     # Useful commands cheatsheet
    └── CODEBASE_REORGANIZATION.md # File organization details
```

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pre-commit install
```

### Training

**Online PPO (JAX):**
```bash
python online_rl/ppo.py --env_name Craftax-Symbolic-v1 --total_timesteps 1e9
```

**Offline AWR Baseline (PyTorch):**
```bash
sbatch scripts/shell/awr_baseline.sh
```

**Offline AWR with VLM (PyTorch):**
```bash
sbatch scripts/shell/run_augmented_awr.sh
```

### Evaluation

**Baseline:**
```bash
python offline_rl/eval_awr.py --checkpoint <path> --num_episodes 5
```

**Augmented (requires VLM server):**
```bash
# Start server
sbatch scripts/shell/submit_vlm_server.sh
# Run eval (check logs for hostname)
python offline_rl/eval_awr_augmented.py --checkpoint <path> --server_url http://<hostname>:5000
```

---

## Key Data Paths

All data is under `/data/group_data/rl/geney/`:

| Path | Purpose |
|------|---------|
| `craftax_unlabelled_new/` | Raw trajectories from PPO |
| `craftax_labelled_results/` | Trajectories with LLM annotations |
| `craftax_labelled_results_with_returns/` | Complete training data |
| `checkpoints/awr_baseline_v2/` | Baseline AWR checkpoints |
| `checkpoints/awr_augmented/` | Augmented AWR checkpoints + stats |
| `craftax_job_logs/` | SLURM job logs |

---

## Labelling Pipeline

The distributed labelling system uses **Redis** for job coordination and **vLLM** for fast inference:

```
PPO Training → Unlabelled NPZ → Redis Queue → GPU Workers (vLLM) → Labelled NPZ (with hidden_state)
```

**Key Scripts:**
- `labelling/llm_worker.py` - Main vLLM-based worker (extracts last-token hidden states)
- `labelling/addtoqueue_llm.py` - Add files to Redis queue
- `labelling/janitor_llm.py` - Re-queue failed jobs
- `labelling/obs_to_text.py` - Decode symbolic observations to text
- `labelling/add_text_obs.py` - Add text_obs to existing NPZ files

**SBATCH Scripts:**
- `labelling/run_labelling.sbatch` - End-to-end: Redis + queue + workers + monitoring
- `labelling/run_extract_hidden.sbatch` - Re-extract hidden states for existing data
- `labelling/makeworkers_llm.sbatch` - Spawn GPU workers (each starts own vLLM server)

See [docs/COMMANDS_REFERENCE.md](docs/COMMANDS_REFERENCE.md) for common commands.

---

## VLM Server Architecture

For **augmented evaluation**, observations are sent to a VLM server that returns hidden states:

```
Observation → VLM Server (Qwen3-VL-4B) → Hidden State (2560-dim) → Policy
```

**Server Endpoints:**
- `GET /health` - Check server status
- `POST /get_hidden_state` - Get hidden state from observation

**Hidden State Processing:**
1. Generate 256 tokens with VLM
2. Extract last-layer hidden states
3. Mean pool across sequence → (2560,) vector

---

## Model Architectures

**JAX Models** (`models/`):
- `ActorCritic` - MLP for symbolic observations
- `ActorCriticConv` - CNN for pixel observations
- `ICMNetwork` - Intrinsic Curiosity Module
- `RNDNetwork` - Random Network Distillation

**PyTorch Models** (inline in `offline_rl/`):
- `ActorCriticConv` - CNN encoder + actor/critic heads
- `ActorCriticConvAug` - Adds 2560-dim hidden state input

---

## Common Flags

### PPO (Online RL)
- `--env_name` - Environment (default: Craftax-Symbolic-v1)
- `--total_timesteps` - Training steps (default: 1e9)
- `--num_envs` - Parallel environments (default: 8192)
- `--save_trajectory` - Save data for offline RL
- `--train_icm` - Enable curiosity-driven exploration

### AWR (Offline RL)
Key config in `Config` class:
- `DATA_DIR` - Path to training data
- `BATCH_SIZE` - Training batch size (default: 256)
- `AWR_BETA` - Temperature parameter (default: 0.1-1.0)

---

## Development Practices

**CRITICAL: Make changes reversible before destructive operations.**

```bash
# Always commit first
git add -A && git commit -m "Checkpoint before changes"

# For experiments, use a branch
git checkout -b experiment/feature-name

# To recover deleted files
git checkout HEAD -- path/to/file
```

See [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for detailed practices.

---

## Important Files Reference

| Category | Files |
|----------|-------|
| **Online RL** | `online_rl/ppo.py`, `ppo_rnd.py`, `ppo_rnn.py` |
| **Offline RL** | `offline_rl/awr.py`, `awr_augmented.py`, `bc.py` |
| **Evaluation** | `offline_rl/eval_awr.py`, `eval_awr_augmented.py` |
| **VLM Server** | `servers/vlm_server.py` |
| **LLM Agents** | `llm_play/llm_play_harnessed.py` |
| **Data Utils** | `tools/compute_hidden_stats.py`, `ppo_add_returns.py` |
| **Wrappers** | `utils/wrappers.py` |

---

## Troubleshooting

**VLM Server Issues:**
```bash
# Check if server is running
curl http://hostname:5000/health

# Server takes ~2 min to load model
# Check logs for "Server ready!"
```

**CUDA/JAX Issues:**
```bash
# Force CPU for evaluation (VLM uses GPU)
JAX_PLATFORM_NAME=cpu python offline_rl/eval_awr_augmented.py ...
```

**Redis Queue Issues:**
```bash
# Check queue length
cd ~/redis-stable; ./src/redis-cli -h $(cat /data/group_data/rl/geney/redis_host.txt) llen craftax_llm_job_queue

# Run janitor to re-queue failed jobs
python labelling/janitor_llm.py
```

---

## Related Documentation

- [docs/progress_journal.md](docs/progress_journal.md) - **Running experiment log & TODO roadmap**
- [docs/llm_craftax_prompting.md](docs/llm_craftax_prompting.md) - LLM player prompts
- [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) - Detailed development guide
- [docs/HIDDEN_STATE_PIPELINE.md](docs/HIDDEN_STATE_PIPELINE.md) - Hidden state extraction
- [docs/COMMANDS_REFERENCE.md](docs/COMMANDS_REFERENCE.md) - Useful commands cheatsheet
- [docs/llm_labelling.md](docs/llm_labelling.md) - LLM data labelling pipeline

