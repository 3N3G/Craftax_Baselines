# LLM Data Labelling System

Complete guide for generating and labelling Craftax trajectory data using LLM hidden states.

---

## Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ run_ppo_symbolic│ --> │ addtoqueue_llm   │ --> │ llm_worker          │
│ (generate data) │     │ (queue to Redis) │     │ (extract hidden st) │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
```

---

## Files

| File | Purpose |
|------|---------|
| `run_ppo_symbolic.sbatch` | Generate symbolic trajectory data |
| `labelling/llm_worker.py` | Process trajectories through LLM |
| `labelling/addtoqueue_llm.py` | Queue files to Redis |
| `labelling/makeworkers_llm.sbatch` | Launch worker array |

---

## Step 1: Generate Data

```bash
ssh babel
cd /home/geney/Craftax_Baselines
sbatch run_ppo_symbolic.sbatch
```

**Output:** `/data/group_data/rl/geney/craftax_unlabelled_symbolic/`

---

## Step 2: Start Redis

```bash
redis-server --daemonize yes --port 6379
```

---

## Step 3: Queue Jobs

```bash
cd /home/geney/Craftax_Baselines/labelling

# --name is optional (data goes to root dir)
python addtoqueue_llm.py --host login1 --symbolic

# Or with subdirectory
python addtoqueue_llm.py --host login1 --symbolic --name gene
```

---

## Step 4: Run Workers

```bash
sbatch makeworkers_llm.sbatch
```

---

## Step 5: Monitor

```bash
squeue -u geney
tail -f /data/group_data/rl/geney/craftax_llm_job_logs/worker_*.log
```

---

## Output Format

Labelled NPZ files contain:
- Original: `obs`, `next_obs`, `action`, `reward`, `done`, `log_prob`
- Added: `hidden_state` (float16), `text_generated`

---

## Configuration

| Setting | Value |
|---------|-------|
| Queue | `craftax_llm_job_queue` |
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| Tokens | 256 |
| dtype | float16 |

---

## Rendering

See [symbolic_to_pixels.md](symbolic_to_pixels.md) for converting symbolic observations to pixel renders for evaluation.
