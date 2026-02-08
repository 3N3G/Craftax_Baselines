# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Craftax Baselines is a research repository for training RL agents on the Craftax environment (a procedurally-generated Minecraft-inspired gridworld). The codebase contains both **online RL** methods (JAX-based) and **offline RL** methods (PyTorch-based), along with distributed infrastructure for trajectory data generation and labelling.

Paper: https://arxiv.org/abs/2402.16801
Main Craftax Repository: https://github.com/MichaelTMatthews/Craftax/

## Installation

```bash
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pre-commit install
```

## Development Practices - Making Changes Reversible

**CRITICAL: Always make changes reversible before performing destructive operations.**

### Git Workflow for Safe Changes

**Before deleting or modifying multiple files:**

1. **Commit current state to git:**
```bash
git add -A
git commit -m "Checkpoint before cleanup/refactor"
```

2. **For experimental changes, use a branch:**
```bash
git checkout -b experiment/cleanup-scripts
# Make changes
git add -A
git commit -m "Experimental cleanup"
# If good: git checkout main && git merge experiment/cleanup-scripts
# If bad: git checkout main && git branch -D experiment/cleanup-scripts
```

3. **For quick experiments, use git stash:**
```bash
# Save current changes
git stash push -m "Trying cleanup approach 1"
# Make changes
# If bad: git stash pop  # Restore previous state
# If good: git stash drop  # Keep current state
```

### Recovering Deleted Files

**If files were accidentally deleted but were previously committed:**

```bash
# See what was deleted
git status

# Restore a specific deleted file
git checkout HEAD -- path/to/file.py

# Restore all deleted files
git checkout HEAD -- .

# Find a file deleted in previous commits
git log --all --full-history --diff-filter=D -- path/to/file.py
git checkout <commit-hash>^ -- path/to/file.py
```

**If files were never committed (check these locations):**

1. Check WandB run directories: `find ./wandb -name "filename.py"`
2. Check backup directories: `find /data/group_data -name "filename.py"`
3. Check SLURM logs: Old command lines may show file contents or usage patterns
4. Check editor backups: `.filename.py~` or `.filename.py.swp`

### Safe File Deletion Checklist

**Before running `rm` commands, always:**

1. ✓ Commit current state to git
2. ✓ Verify files are tracked: `git ls-files | grep filename`
3. ✓ Check if files are used by other scripts: `grep -r "filename" *.sh *.py`
4. ✓ Use `git rm` instead of `rm` when possible (auto-stages deletion)
5. ✓ For testing, move to a backup directory first:
   ```bash
   mkdir .backup_$(date +%Y%m%d)
   mv file1.py file2.sh .backup_*/
   # Test that everything still works
   # If OK: rm -rf .backup_*
   # If broken: mv .backup_*/* .
   ```

### Pre-Commit Hooks

This repo uses pre-commit hooks. To ensure code quality:

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

### Version Control Best Practices

1. **Commit frequently with descriptive messages:**
   ```bash
   git commit -m "Add VLM server for augmented evaluation"
   git commit -m "Fix normalization stats path in eval script"
   ```

2. **Use .gitignore for generated files:**
   - Checkpoints (*.pth, *.pkl)
   - Logs (logs/*, wandb/*)
   - Data files (*.npz, *.h5)
   - Videos (eval_videos/*, *.mp4)

3. **Never commit sensitive data:**
   - API keys (use environment variables)
   - Large model files (use git-lfs or external storage)
   - Personal data or credentials

4. **Review changes before committing:**
   ```bash
   git diff                    # See unstaged changes
   git diff --staged          # See staged changes
   git status                 # Overview of all changes
   ```

### Emergency Recovery

**If you accidentally deleted files without git:**

1. **Don't panic - files may still be recoverable**
2. Check if files are in trash: `ls ~/.local/share/Trash/files/`
3. Search entire filesystem: `find / -name "filename.py" 2>/dev/null`
4. Check if process is still running with deleted file open: `lsof | grep deleted`
5. Restore from backups if available: Check `/data/group_data` or NFS snapshots

**If changes broke the codebase:**

```bash
# Reset to last commit (CAREFUL: loses uncommitted changes)
git reset --hard HEAD

# Reset to specific commit
git reset --hard <commit-hash>

# Restore specific files without losing other changes
git checkout HEAD -- file1.py file2.sh
```

## Quick Start

**Train Baseline (CNN only):**
```bash
sbatch awr_baseline.sh
```

**Train Augmented (CNN + VLM):**
```bash
sbatch run_augmented_awr.sh
```

**Evaluate Baseline:**
```bash
sbatch eval_batch_baseline.sh
```

**Evaluate Augmented:**
```bash
# 1. Start VLM server
sbatch submit_vlm_server.sh
# 2. Wait for server to load (~2 min), get hostname from logs
# 3. Run evaluation
sbatch eval_batch_augmented.sh <hostname>
```

## Running Experiments

### Online RL (JAX-based)

All online RL scripts use JAX and the PureJaxRL architecture. They support command-line arguments for configuration.

```bash
# Standard PPO
python ppo.py

# PPO with RNN
python ppo_rnn.py

# PPO with ICM (curiosity-driven exploration)
python ppo.py --train_icm

# PPO with E3B (epistemic bonus)
python ppo.py --train_icm --use_e3b --icm_reward_coeff 0

# PPO with RND (Random Network Distillation)
python ppo_rnd.py
```

### Offline RL (PyTorch-based)

Recent additions for learning from offline trajectory data:

```bash
# Baseline AWR (CNN only)
python pt_awr.py

# Augmented AWR (CNN + VLM hidden states)
python awr_aug.py

# Behavioral Cloning
python pt_bc.py

# Evaluate baseline AWR models
python pt_eval_awr.py

# Evaluate augmented AWR models (requires VLM server)
python eval_awr_aug_client_proper.py --checkpoint <path> --server_url http://hostname:5000 --stats_path <path>
```

### SLURM Cluster Submission

For SLURM-based execution:

```bash
# Generic training script (pass python file as argument)
sbatch run_train.sh <script.py>

# AWR baseline training
sbatch awr_baseline.sh

# AWR augmented training
sbatch run_augmented_awr.sh

# Baseline evaluation (batch eval of all checkpoints)
sbatch eval_batch_baseline.sh

# Augmented evaluation (requires VLM server, pass hostname as arg)
sbatch submit_vlm_server.sh              # Start VLM server first
sbatch eval_batch_augmented.sh <hostname>  # Then run eval
```

The `run_train.sh` script sets up the conda environment, GPU resources, and WandB configuration for 24-hour jobs.

## Architecture

### Directory Structure

- **Root Python files**: Training scripts (`ppo.py`, `pt_awr.py`, etc.)
- **`models/`**: Neural network architectures
  - Actor-Critic networks (MLP and Conv variants)
  - ICM (Intrinsic Curiosity Module) components
  - RND (Random Network Distillation) networks
- **`wrappers.py`**: Environment wrappers for Craftax
  - `LogWrapper`: Episode return/length tracking
  - `AutoResetEnvWrapper`: Standard auto-reset on episode termination
  - `OptimisticResetVecEnvWrapper`: Efficient vectorized resets with reset pooling
  - `BatchEnvWrapper`: Vectorizes environment operations
- **`logz/`**: Logging utilities with WandB integration
- **`labelling/`**: Distributed trajectory labelling infrastructure
  - `janitor.py`: Redis-based job queue monitor/manager
  - Worker scripts for SLURM-based distributed processing
- **`analysis/`**: Analysis and visualization tools
- **`logs/`**: SLURM job logs
- **`wandb/`**: WandB run artifacts

### Complete File Reference

#### Training Scripts (SLURM)
- **`awr_baseline.sh`** - Train baseline AWR (CNN only, no VLM)
- **`run_augmented_awr.sh`** - Train augmented AWR (CNN + VLM hidden states)
- **`run_train.sh`** - Generic SLURM wrapper for any Python training script

#### Evaluation Scripts (SLURM)
- **`eval_batch_baseline.sh`** - Batch evaluate all baseline checkpoints (25k, 50k, 75k, 100k)
- **`eval_batch_augmented.sh`** - Batch evaluate all augmented checkpoints (requires VLM server)
- **`submit_vlm_server.sh`** - Start VLM server on SLURM GPU node

#### Core Training (Python)
- **`ppo.py`** - Online PPO training (JAX) with optional ICM/E3B exploration
- **`ppo_rnn.py`** - PPO with recurrent networks
- **`ppo_rnd.py`** - PPO with Random Network Distillation exploration
- **`ppo_pixel.py`** - PPO for pixel observations
- **`pt_awr.py`** - Offline AWR baseline (PyTorch, CNN only)
- **`awr_aug.py`** - Offline AWR augmented (PyTorch, CNN + VLM hidden states)
- **`pt_bc.py`** - Behavioral cloning from offline data

#### Core Evaluation (Python)
- **`pt_eval_awr.py`** - Evaluate baseline AWR checkpoints (standalone)
- **`eval_awr_aug_client_proper.py`** - Evaluate augmented AWR checkpoints (client, connects to VLM server)
- **`vlm_server.py`** - VLM server hosting Qwen3-VL-4B model, provides HTTP API for hidden state generation

#### Data Processing & Utilities
- **`ppo_add_returns.py`** - Add Monte Carlo returns to trajectory data (post-processing)
- **`collectdatasetstats.py`** - Compute dataset statistics (episode returns, lengths, etc.)
- **`compute_hidden_stats.py`** - Compute normalization statistics (mean/std) for VLM hidden states from training data
- **`wrappers.py`** - Environment wrappers (LogWrapper, AutoReset, OptimisticReset, etc.)
- **`image_utils.py`** - Image format conversion utilities (0-1 ↔ 0-255, numpy ↔ PIL)
- **`diagnose_hidden_state_mismatch.py`** - Diagnostic tool to compare training vs server hidden states

#### Infrastructure
- **`build.sh`** - Build Docker container
- **`run_docker.sh`** - Run code in Docker container
- **`CLAUDE.md`** - This file (project documentation for Claude Code)
- **`README.md`** - General project readme

### Key Architectural Patterns

#### Online RL Training Loop (JAX)

The online RL scripts (`ppo.py`, `ppo_rnn.py`, `ppo_rnd.py`) follow the PureJaxRL pattern:
1. Environment wrapped with `LogWrapper` → `OptimisticResetVecEnvWrapper` or (`AutoResetEnvWrapper` → `BatchEnvWrapper`)
2. JAX-based vectorized rollouts with `jax.vmap` and `jax.lax.scan`
3. PPO updates with minibatch SGD
4. Optional intrinsic reward computation (ICM, E3B, or RND)
5. Checkpoint saving via Orbax
6. WandB logging via `logz.batch_logging`

Key configuration parameters:
- `NUM_ENVS`: Parallel environments (typically 128)
- `NUM_STEPS`: Rollout length
- `NUM_MINIBATCHES`: Minibatch count for PPO updates
- `UPDATE_EPOCHS`: PPO update epochs per rollout
- `USE_OPTIMISTIC_RESETS`: Enable optimistic reset wrapper for efficiency

#### Offline RL Training (PyTorch)

The offline RL scripts (`pt_awr.py`, `pt_bc.py`, `awr_aug.py`) follow a different pattern:
1. Load pre-collected trajectory data from `.npz` files
2. PyTorch-based actor-critic networks with CNN encoders
3. AWR: Advantage-weighted policy updates with exponential weighting
4. BC: Supervised learning with cross-entropy loss
5. Evaluation via JAX-based environment rollouts (hybrid JAX/PyTorch)
6. Model checkpoints saved periodically

Data format (`.npz` files):
- `obs`: Observations (images or symbolic state)
- `next_obs`: Next observations
- `action`: Actions taken
- `reward`: Rewards received
- `done`: Episode termination flags
- `log_prob`: (optional) Action log probabilities from behavior policy
- `hidden_state`: (augmented only) VLM hidden states (80, 2560) from Qwen3-VL model

#### VLM Augmentation System (Client-Server Architecture)

The augmented AWR model uses **Vision Language Model (VLM) hidden states** as additional input to improve policy learning. This requires a client-server architecture:

**Training (`awr_aug.py`):**
- Loads pre-computed VLM hidden states from training data
- Model: `ActorCriticConvAug` - CNN encoder + hidden state fusion
- Hidden states are normalized using dataset statistics (mean/std)
- Input: Observation (130, 110, 3) + Hidden state (2560,)
- Output: Action distribution + Value estimate

**Evaluation (Client-Server):**
1. **VLM Server** (`vlm_server.py`):
   - Hosts Qwen3-VL-4B-Instruct model on GPU
   - Flask HTTP API on port 5000
   - Endpoints:
     - `GET /health` - Check server status
     - `POST /get_hidden_state` - Generate hidden state from observation
   - Processing: Image → VLM forward pass → Extract last layer hidden states → Subsample every 8 tokens → Mean pool → (2560,) vector
   - Takes ~1-2s per inference (VLM is slow)

2. **Evaluation Client** (`eval_awr_aug_client_proper.py`):
   - Loads trained policy checkpoint
   - Connects to VLM server via HTTP requests
   - For each step: Send observation → Receive hidden state → Normalize → Policy forward → Get action
   - Saves videos with dual-line visualization (Value function + Return-to-Go)
   - Logs metrics to WandB

**Normalization Statistics:**
- Location: `/data/group_data/rl/geney/checkpoints/awr_augmented/hidden_state_stats.npz`
- Contains: `mean` (2560,), `std` (2560,) computed from training data
- Critical: Evaluation must use the same normalization as training
- Regenerate with: `python compute_hidden_stats.py --data_dir <path> --output <path>`

**Why Client-Server?**
- VLM is large (4B parameters) and requires GPU
- Server can handle multiple evaluation jobs
- Avoids loading VLM model multiple times
- Separates fast policy execution (CPU) from slow VLM inference (GPU)

#### Data Generation & Labelling Workflow

For offline RL, trajectories are generated and labelled in a distributed manner:

1. **Generation**: PPO training with `--save_trajectory` flag saves rollout data to `/data/group_data/rl/craftax_unlabelled_new/`
2. **Labelling**:
   - `janitor.py` monitors unlabelled files and maintains a Redis job queue
   - SLURM worker jobs pull tasks from the queue and process trajectories
   - `ppo_add_returns.py` adds Monte Carlo returns to trajectories
   - Labelled data saved to `/data/group_data/rl/craftax_labelled_results_with_returns/`
3. **Analysis**: `collectdatasetstats.py` computes dataset statistics (episode returns, lengths)

The Redis queue ensures fault tolerance - if workers die, `janitor.py` re-enqueues orphaned jobs.

### Environment Wrappers

**OptimisticResetVecEnvWrapper** is the recommended choice for training:
- Resets environments in batches with a configurable `reset_ratio` (e.g., 128 envs with reset_ratio=4 means 32 reset operations)
- When episodes terminate, randomly selects from pre-computed resets
- Significantly faster than per-environment resets
- Trade-off: Slight chance of duplicate resets

**AutoResetEnvWrapper + BatchEnvWrapper** is the fallback:
- Guaranteed unique resets per environment
- Slower due to per-environment reset overhead

### Model Architectures

**JAX models** (`models/actor_critic.py`, `models/icm.py`, `models/rnd.py`):
- Built with Flax (JAX neural network library)
- `ActorCritic`: MLP-based for symbolic observations
- `ActorCriticConv`: CNN-based for pixel observations
- Orthogonal initialization for stable training

**PyTorch models** (defined inline in `pt_*.py` files):
- `ActorCriticConv`: CNN encoder (3 conv layers + pooling) → Actor/Critic heads
- `ActorCriticConvAug`: Adds hidden state augmentation (concatenates CNN features with 2560-dim hidden states)
- Orthogonal initialization matching JAX models

## Commonly Used Flags

### PPO Flags
- `--ENV_NAME`: Environment name (default: "Craftax-Symbolic-v1")
- `--TOTAL_TIMESTEPS`: Total training steps (default: 1e9)
- `--NUM_ENVS`: Parallel environments (default: 8192 for symbolic, 4096 for pixels)
- `--NUM_STEPS`: Rollout length (default: 64)
- `--NUM_MINIBATCHES`: Minibatch count (default: 16)
- `--UPDATE_EPOCHS`: PPO epochs per rollout (default: 4)
- `--LR`: Learning rate (default: 2e-4)
- `--ANNEAL_LR`: Anneal learning rate (default: True)
- `--MAX_GRAD_NORM`: Gradient clipping (default: 1.0)
- `--USE_OPTIMISTIC_RESETS`: Enable optimistic resets (default: True)
- `--OPTIMISTIC_RESET_RATIO`: Reset ratio (default: 64)
- `--train_icm`: Enable ICM intrinsic motivation
- `--use_e3b`: Enable E3B (requires --train_icm)
- `--save_policy`: Save trained policy to disk
- `--save_trajectory`: Save trajectory data for offline RL

### Offline RL Configuration

Configurations are typically defined in a `Config` class at the top of each script. Key parameters:
- `DATA_DIR`: Path to trajectory data
- `BATCH_SIZE`: Training batch size (default: 256)
- `LR`: Learning rate (default: 3e-4)
- `TOTAL_STEPS`: Training steps (default: 100k)
- `GAMMA`: Discount factor (default: 0.99)
- `AWR_BETA`: AWR temperature parameter (default: 0.1-1.0)
- `AWR_MAX_WEIGHT`: Maximum advantage weight (default: 20.0)

## WandB Integration

All training scripts log to Weights & Biases (WandB). Set the API key:
```bash
export WANDB_API_KEY="your-key-here"
```

Logged metrics include:
- Episode returns and lengths
- Loss values (policy, value, entropy)
- Learning rates
- Gradient norms
- Intrinsic/extrinsic reward statistics

## Data Paths

**Important**: All data paths are now under `/data/group_data/rl/geney/`:
- Unlabelled trajectories: `/data/group_data/rl/geney/craftax_unlabelled_new/{name}/`
- Labelled trajectories: `/data/group_data/rl/geney/craftax_labelled_results/{name}/`
- Labelled with returns: `/data/group_data/rl/geney/craftax_labelled_results_with_returns/`
- Checkpoints: `/data/group_data/rl/geney/checkpoints/`
- Job logs: `/data/group_data/rl/geney/craftax_job_logs/`

Update these paths in the `Config` classes when running locally or on different infrastructure.

## Redis Configuration (Labelling System)

The distributed labelling system uses Redis for job queuing:
- Queue name: `craftax_job_queue`
- Scripts require `--host` (login node) and `--port` arguments
- `janitor.py` monitors for orphaned jobs and re-queues them every 5 minutes

## Checkpointing

### JAX Models (Orbax)
Checkpoints saved via Orbax CheckpointManager to the path specified in `--save_dir` flag. Can be loaded for evaluation or fine-tuning.

### PyTorch Models
Checkpoints saved via `torch.save()` at intervals defined by `SAVE_FREQ`. Saved to `Config.SAVE_DIR`.

**Checkpoint Locations:**
- Baseline AWR: `/data/group_data/rl/geney/checkpoints/awr_baseline_v2/`
  - `awr_checkpoint_25000.pth`, `awr_checkpoint_50000.pth`, `awr_checkpoint_75000.pth`, `awr_checkpoint_100000.pth`
- Augmented AWR: `/data/group_data/rl/geney/checkpoints/awr_augmented/`
  - `awr_aug_checkpoint_25000.pth`, `awr_aug_checkpoint_50000.pth`, `awr_aug_checkpoint_75000.pth`, `awr_aug_checkpoint_100000.pth`
  - `hidden_state_stats.npz` - Normalization statistics for hidden states (required for evaluation)

## Complete Workflow

### 1. Train Baseline AWR Model

```bash
sbatch awr_baseline.sh
```

This trains a baseline AWR policy using only CNN observations (no VLM). Checkpoints saved every 25k steps to `/data/group_data/rl/geney/checkpoints/awr_baseline_v2/`.

### 2. Train Augmented AWR Model

```bash
sbatch run_augmented_awr.sh
```

This trains an augmented AWR policy using CNN observations + VLM hidden states. Requires pre-computed hidden states in training data. Checkpoints saved every 25k steps to `/data/group_data/rl/geney/checkpoints/awr_augmented/`.

### 3. Evaluate Baseline Model

```bash
sbatch eval_batch_baseline.sh
```

Evaluates all baseline checkpoints (25k, 50k, 75k, 100k) sequentially. Each checkpoint runs 3 episodes. Videos and metrics logged to WandB.

### 4. Evaluate Augmented Model

**Step 1: Start VLM Server**
```bash
sbatch submit_vlm_server.sh
```

This starts the Qwen3-VL server on a GPU node. Check logs to get the hostname (e.g., `babel-v5-16`). Server takes ~1-2 minutes to load the model.

**Step 2: Run Evaluation**
```bash
sbatch eval_batch_augmented.sh babel-v5-16
```

Replace `babel-v5-16` with the actual hostname from Step 1. This evaluates all augmented checkpoints, sending observations to the VLM server to get hidden states.

### 5. Compute Hidden State Normalization Stats (if needed)

If you need to regenerate normalization statistics:

```bash
python compute_hidden_stats.py \
    --data_dir /data/group_data/rl/geney/craftax_labelled_results_with_returns \
    --output /data/group_data/rl/geney/checkpoints/awr_augmented/hidden_state_stats.npz \
    --num_files 10
```

This computes mean and std of hidden states from training data. Required for augmented model evaluation.

## VLM Server API Reference

### Starting the Server

**Manual Start:**
```bash
python vlm_server.py --host 0.0.0.0 --port 5000
```

**SLURM Start:**
```bash
sbatch submit_vlm_server.sh
```

Server takes ~1-2 minutes to load Qwen3-VL-4B model. Check logs to confirm it's ready.

### Endpoints

**Health Check:**
```bash
curl http://hostname:5000/health
```

Response:
```json
{
  "status": "ready",
  "model": "Qwen/Qwen3-VL-4B-Instruct"
}
```

**Get Hidden State:**
```bash
curl -X POST http://hostname:5000/get_hidden_state \
  -H "Content-Type: application/json" \
  -d '{"obs": [[...]]}'  # Observation as nested list (H, W, C)
```

Response:
```json
{
  "hidden_state": [float, ...],  # 2560 floats
  "shape": [2560]
}
```

### Hidden State Processing Pipeline

1. Receive observation (H, W, C) as numpy array in 0-1 range
2. Convert to PIL Image (scale to 0-255)
3. Create prompt with game description + question
4. Process through Qwen3-VL model (generate 256 tokens)
5. Concatenate hidden states from prompt + generated tokens (seq_len ≈ 633)
6. Subsample every 8th token from the end: `indices = range(seq_len-1, -1, -8)`
7. Result: ~80 subsampled tokens (matches training data)
8. Mean pool across sequence dimension (80, 2560) → (2560,)
9. Return as JSON

**Important**: Server returns **unnormalized** hidden states. Client must normalize using training statistics before feeding to policy.

**Critical**: The server processing must exactly match `labelling/run_worker.py` which generated the training data. Key settings:
- `TOKENS_GENERATED = 256` (generates 256 new tokens)
- Hidden states include both prompt (~377 tokens) and generated (256 tokens) = ~633 total
- Subsample every 8 from end → ~80 tokens (matches training data shape `(N, 80, 2560)`)

## Recent Fixes (January 2026)

### VLM Server Token Sampling Mismatch (Fixed)

**Problem**: Evaluation hidden states didn't match training data distribution, causing poor policy performance.

**Root Cause**: The VLM server was configured with `TOKENS_GENERATED = 640`, but training data (`labelling/run_worker.py`) used `TOKENS_GENERATED = 256`. This caused:
- Training: ~80 subsampled tokens (from ~633 total = 377 prompt + 256 generated)
- Eval (broken): ~105 subsampled tokens (from ~840 total = 377 prompt + 640 generated)

**Fix**: Changed `vlm_server.py` to use `TOKENS_GENERATED = 256` to match training.

**Verification**: Run `diagnose_hidden_state_mismatch.py` and check:
- Subsampled tokens: should be ~80 (matching training)
- Normalized L2 norm ratio: should be ~0.93+ (close to 1.0)

### Server Connectivity from Different Nodes

**Problem**: VLM server only accessible from the same node.

**Root Cause**: Using `127.0.0.1` (localhost) instead of actual IP/hostname.

**Fix**: When connecting from a different node, use:
- The IP address shown in server startup (e.g., `http://10.1.1.75:5000`)
- Or the hostname (e.g., `http://babel-p9-16:5000`)

NOT `http://127.0.0.1:5000` which only works on the same machine.

---

## Troubleshooting

### VLM Server Issues

**"Server not ready"**
- Check if server is running: `curl http://hostname:5000/health`
- Server logs: Check SLURM output file `logs/vlm_server_*.out`
- Model loading takes ~1-2 minutes, be patient

**"Connection refused"**
- Verify hostname is correct (get from SLURM logs)
- Ensure firewall allows port 5000
- Try with IP address instead of hostname

**Out of GPU memory**
- VLM server requires ~20GB GPU memory
- Use a GPU node with sufficient memory
- Close other GPU processes

### Evaluation Issues

**"Normalization stats not found"**
- Check file exists: `ls -la /data/group_data/rl/geney/checkpoints/awr_augmented/hidden_state_stats.npz`
- Regenerate with `compute_hidden_stats.py` if missing

**"Model checkpoint not found"**
- Check checkpoint directory exists and contains `.pth` files
- Verify path in eval script matches actual checkpoint location

**CUDA errors during augmented eval**
- Augmented eval should use CPU for JAX (environment simulation)
- VLM server uses GPU for Qwen3-VL
- Check environment variables: `JAX_PLATFORM_NAME=cpu`, `JAX_PLATFORMS=cpu`

**Evaluation is very slow**
- VLM inference takes ~1-2s per step (this is expected)
- 300-step episode = ~10 minutes with VLM
- Baseline eval (no VLM) is much faster (~1 minute per episode)

### Training Issues

**Out of memory during augmented training**
- Reduce `BATCH_SIZE` in `awr_aug.py` Config
- Current default: 256, try 128 or 64

**"Hidden states dimension mismatch"**
- Training data must have `hidden_state` field with shape (N, 80, 2560) or (N, 2560)
- Check data with: `python -c "import numpy as np; print(np.load('file.npz')['hidden_state'].shape)"`

## Docker Support

Dockerfile and `run_docker.sh` provided for containerized execution. Build with:
```bash
bash build.sh
```

## Code Origins

Based on Chris Lu's PureJaxRL implementation: https://github.com/luchris429/purejaxrl

The codebase adapts PureJaxRL for Craftax with additional exploration methods (ICM, RND, E3B) and offline RL extensions.

---

## Important Reminders for Claude Code

**When asked to clean up, refactor, or delete files:**

1. ✓ **ALWAYS commit to git first**: `git add -A && git commit -m "Before cleanup"`
2. ✓ **Check dependencies**: `grep -r "filename" *.sh *.py` before deleting
3. ✓ **Verify files can be recovered**: Ensure files are tracked by git
4. ✓ **Ask for confirmation**: If deleting >5 files, list them and ask user to confirm
5. ✓ **Test incrementally**: Delete/change one category at a time, verify it works
6. ✓ **Document what was deleted**: Keep a record in commit message or temporary file

**Red flags that require extra caution:**
- Deleting files with "server", "client", "eval", "config" in the name
- Removing files that other scripts might import or call
- Bulk deletions (>10 files at once)
- Modifying files in production data directories
- Changes to files that haven't been read yet in the conversation

**Safe alternatives to deletion:**
- Move to `.archive/` directory instead of deleting
- Comment out code instead of removing
- Use git branches for experiments
- Create backup directory with timestamp before cleanup

**If something goes wrong:**
- Check `./wandb/*/files/code/` for backup copies
- Use `git reflog` and `git checkout` to recover
- Search filesystem: `find . -name "filename"`
- User may have backups in `/data/group_data/rl/geney/`
