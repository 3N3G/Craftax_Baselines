# Commands Reference

Quick reference for commonly used commands when working with the Craftax codebase.

---

## Monitoring

### Watch Jobs and Files
```bash
# Watch latest files and job queue
watch -n 1 -c "ls -ltr --color=always | tail -10; echo ''; squeue -u geney"

# Monitor SLURM jobs
squeue -u geney -o "%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R"

# Watch specific output file
tail -f logs/<job_name>_<jobid>.out
```

### Check GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

---

## Redis Queue (Labelling)

### Connect to Redis
```bash
cd ~/redis-stable
./src/redis-cli -h $(cat /data/group_data/rl/geney/redis_host.txt)
```

### Queue Commands
```bash
# Check queue length
cd ~/redis-stable; ./src/redis-cli -h $(cat /data/group_data/rl/geney/redis_host.txt) llen craftax_llm_job_queue

# Peek at queue contents
cd ~/redis-stable; ./src/redis-cli -h $(cat /data/group_data/rl/geney/redis_host.txt) lrange craftax_llm_job_queue 0 10

# Clear queue (DANGEROUS)
cd ~/redis-stable; ./src/redis-cli -h $(cat /data/group_data/rl/geney/redis_host.txt) del craftax_llm_job_queue
```

---

## Labelling Pipeline

### End-to-End Labelling (Recommended)
```bash
# Full pipeline: starts Redis, queues files, spawns workers, monitors
sbatch labelling/run_labelling.sbatch

# Re-extract hidden states for files that already have text_generated
sbatch labelling/run_extract_hidden.sbatch
```

### Manual Control
```bash
# Add unlabelled files to queue
python labelling/addtoqueue_llm.py \
    --host $(cat /data/group_data/rl/geney/redis_host.txt) \
    --symbolic

# Spawn GPU workers (each starts own vLLM server)
sbatch labelling/makeworkers_llm.sbatch

# Re-queue failed jobs
python labelling/janitor_llm.py --host $(cat /data/group_data/rl/geney/redis_host.txt)
```

---

## Training

### Online PPO (JAX)
```bash
# Symbolic environment (fast)
python online_rl/ppo.py --env_name Craftax-Symbolic-v1 --total_timesteps 1e9 --num_envs 8192

# With trajectory saving for offline RL
python online_rl/ppo.py --save_trajectory --save_trajectory_every 100

# Submit to SLURM
sbatch scripts/sbatch/run_ppo_symbolic.sbatch
```

### Offline AWR (PyTorch)
```bash
# Baseline (CNN only)
python offline_rl/awr.py

# Augmented (CNN + VLM hidden states)
python offline_rl/awr_augmented.py

# SLURM submission
sbatch scripts/shell/awr_baseline.sh
sbatch scripts/shell/run_augmented_awr.sh
```

### Online RL with LLM Hidden States
```bash
# Start vLLM server (for LLM mode)
bash scripts/start_vllm_hidden.sh --mode last_token

# JAX trainer (current): LLM every 25 env-steps, prefill hidden (tokens=1)
python online_rl_llm/online_rl_hidden_jax.py \
    --envs 128 --timesteps 100000000 --skip-n 25 --tokens 1 --layer -1

# No-LLM baseline mode in same stack
python online_rl_llm/online_rl_hidden_jax.py \
    --no-llm --envs 128 --timesteps 100000000

# Architecture controls
python online_rl_llm/online_rl_hidden_jax.py \
    --envs 128 --timesteps 10000000 --skip-n 25 \
    --fusion-mode dual_concat --actor-head-layers 2 --critic-head-layers 2

# Periodic checkpoints + resumable state
python online_rl_llm/online_rl_hidden_jax.py \
    --envs 128 --timesteps 100000000 --skip-n 25 \
    --checkpoint-every-steps 10000000 \
    --policy-save-dir /data/group_data/rl/geney/online_rl_hidden_models \
    --checkpoint-dir /data/group_data/rl/geney/online_rl_hidden_models

# Resume from latest checkpoint in a directory
python online_rl_llm/online_rl_hidden_jax.py \
    --resume-from /data/group_data/rl/geney/online_rl_hidden_models \
    --run-name online-jax-128env-skip25-resumed

# SLURM submission (includes optional vLLM startup)
sbatch scripts/sbatch/run_online_rl_hidden_jax.sbatch 128 100000000 25 -1 1 64 10000000
# args: envs timesteps skip_n layer tokens num_steps checkpoint_every_steps

# Clear corrupted vLLM cache if needed
bash scripts/clear_vllm_cache.sh
```

---

## Evaluation

### Symbolic Policy Suite (W&B `craftax_symbolic_evals`)
```bash
# Local run (evaluates skip25, skip100m, PPO by default)
python scripts/eval_symbolic_policy_suite.py \
    --wandb-project craftax_symbolic_evals \
    --wandb-entity iris-sobolmark \
    --num-envs 128 \
    --target-episodes 128

# Cluster run
sbatch scripts/sbatch/run_symbolic_policy_evals.sbatch

# PPO-only rerun (no vLLM needed)
START_VLLM=0 sbatch scripts/sbatch/run_symbolic_policy_evals.sbatch --policies ppo_symbolic
```

### Offline LLM-Augmented Symbolic Eval
```bash
# Direct script
python scripts/eval_offline_llm_symbolic.py \
    --checkpoint /data/group_data/rl/geney/checkpoints/awr_llm_augmented_live/<run>/awr_llm_final.pth \
    --hidden-input-mode llm \
    --num-episodes 128 --num-envs 128

# Cluster wrapper
CHECKPOINT=/data/group_data/rl/geney/checkpoints/awr_llm_augmented_live/<run>/awr_llm_final.pth \
HIDDEN_INPUT_MODE=llm \
NO_WANDB=0 \
sbatch scripts/sbatch/run_eval_offline_llm_symbolic.sbatch
```

---

## VLM Server

### Start Server
```bash
# Manual start
python servers/vlm_server.py --host 0.0.0.0 --port 5000

# SLURM submission
sbatch scripts/shell/submit_vlm_server.sh
```

### Health Check
```bash
curl http://<hostname>:5000/health
```

### Test Hidden State Extraction
```bash
curl -X POST http://<hostname>:5000/get_hidden_state \
    -H "Content-Type: application/json" \
    -d '{"obs": [[[...]]]}' 
```

---

## Data Processing

### Compute Hidden State Statistics
```bash
python tools/compute_hidden_stats.py \
    --data_dir /data/group_data/rl/geney/craftax_labelled_results_with_returns \
    --output /data/group_data/rl/geney/checkpoints/awr_augmented/hidden_state_stats.npz \
    --num_files 100
```

### Add Returns to Trajectories
```bash
# Exact RTG recomputation over labelled data
python tools/ppo_add_returns_exact.py \
    --input_dir /data/group_data/rl/geney/vllm_craftax_labelled_results \
    --output_dir /data/group_data/rl/geney/vllm_craftax_labelled_results

# Legacy path (writes to separate output dir)
python tools/ppo_add_returns.py \
    --input_dir /data/group_data/rl/geney/craftax_labelled_results \
    --output_dir /data/group_data/rl/geney/craftax_labelled_results_with_returns
```

### Collect Dataset Statistics
```bash
python tools/collectdatasetstats.py \
    --data_dir /data/group_data/rl/geney/craftax_labelled_results_with_returns
```

### Value Learning / Probe Diagnostics
```bash
# Dataset-level value-vs-RTG metrics for one or more checkpoints
python offline_rl/analyze_value_learning.py \
    --checkpoints /data/group_data/rl/geney/checkpoints/awr_llm_augmented_live/<run>/awr_llm_final.pth \
    --dataset_dir /data/group_data/rl/geney/vllm_craftax_labelled_results \
    --num_samples 20000 \
    --hidden_mode real \
    --output_json logs/value_learning_<tag>.json

# Generate paired probe states
python offline_rl/generate_value_probe_pairs.py \
    --output_dir analysis/value_probe_pairs_seed42_smoke \
    --num_states 64 --seed 42

# Pairwise monotonicity checks (zero/random/llm_no_cot hidden modes)
python offline_rl/analyze_value_pairs.py \
    --checkpoints /data/group_data/rl/geney/checkpoints/awr_llm_augmented_live/<run>/awr_llm_final.pth \
    --hidden_input_mode llm_no_cot \
    --output_json logs/value_pairs_<tag>.json
```

---

## Human Play / Recording

### Record Gameplay
```bash
python tools/play_craftax_recorder.py
```

### Query LLM Interactively
```bash
python llm_play/query_qwen.py
```

---

## Git / Version Control

### Checkpoint Before Changes
```bash
git add -A && git commit -m "Checkpoint before changes"
```

### Recover Deleted Files
```bash
git checkout HEAD -- path/to/file
```

### Check Status
```bash
git status
git diff --staged
```

---

## File Operations

### Find Large Files
```bash
find /data/group_data/rl/geney -name "*.npz" -size +100M | head -20
```

### Count Labelled Files
```bash
ls /data/group_data/rl/geney/craftax_labelled_results_with_returns/*.npz | wc -l
```

### Check NPZ Contents
```bash
python -c "import numpy as np; d = np.load('file.npz'); print(list(d.keys())); print({k: d[k].shape for k in d.keys()})"
```

---

## Environment Setup

### Activate Conda Environment
```bash
conda activate craftax
# or
source activate craftax
```

### Check JAX GPU
```bash
python -c "import jax; print(jax.devices())"
```

### Force CPU for JAX (when GPU is used by VLM)
```bash
JAX_PLATFORM_NAME=cpu python offline_rl/eval_awr_augmented.py ...
```
