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
# Start vLLM server first
bash scripts/start_vllm_hidden.sh --mode last_token

# Basic run with prompt-only mode (fast)
python online_rl_llm/online_rl_hidden.py --envs 128 --steps 100000 --skip-n 25

# With token generation (slower, allows reasoning)
python online_rl_llm/online_rl_hidden.py --envs 128 --steps 100000 --skip-n 25 --tokens 64

# Use specific layer (default is -1 for last)
python online_rl_llm/online_rl_hidden.py --envs 128 --steps 100000 --skip-n 25 --layer 24

# SLURM submission (includes vLLM server startup)
sbatch scripts/sbatch/run_online_rl_hidden.sbatch 128 100000000 25  # envs steps skip_n
sbatch scripts/sbatch/run_online_rl_hidden.sbatch 128 100000000 25 -1 64  # with layer and tokens

# Clear corrupted vLLM cache if needed
bash scripts/clear_vllm_cache.sh
```

---

## Evaluation

### Baseline AWR
```bash
python offline_rl/eval_awr.py \
    --checkpoint /data/group_data/rl/geney/checkpoints/awr_baseline_v2/awr_checkpoint_100000.pth \
    --num_episodes 5
```

### Augmented AWR (requires VLM server)
```bash
# 1. Start VLM server
sbatch scripts/shell/submit_vlm_server.sh

# 2. Check logs for hostname (wait ~2 min for model load)
tail -f logs/vlm_server_*.out

# 3. Run evaluation
python offline_rl/eval_awr_augmented.py \
    --checkpoint /data/group_data/rl/geney/checkpoints/awr_augmented/awr_aug_checkpoint_100000.pth \
    --server_url http://<hostname>:5000 \
    --stats_path /data/group_data/rl/geney/checkpoints/awr_augmented/hidden_state_stats.npz \
    --save_video
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
python tools/ppo_add_returns.py \
    --input_dir /data/group_data/rl/geney/craftax_labelled_results \
    --output_dir /data/group_data/rl/geney/craftax_labelled_results_with_returns
```

### Collect Dataset Statistics
```bash
python tools/collectdatasetstats.py \
    --data_dir /data/group_data/rl/geney/craftax_labelled_results_with_returns
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
