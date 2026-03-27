# Codex Agent Handoff (2026-03-10)

Purpose: let a new Codex agent continue week-long online RL + CoT work without re-discovering operational details.

## Read Order For New Agents
1. `AGENTS.md` (Babel safety + command protocol)
2. `docs/progress_journal.md` (latest stage context)
3. this file

## Current Active CoT Runs (Snapshot: 2026-03-10)

These are currently running on Babel (`rl` partition):

| Job ID | Prompt Variant | Run Name | Latest Seen Progress | Checkpoint Dir |
|---|---|---|---|---|
| `6523164` | `future_based` | `online-cot-more-future-based-s25-tok256-stage60m-20260309_152845` | `1,884,160` steps (`Update 230/7324`) | `/data/group_data/rl/geney/online_rl_hidden_models/online-cot-more-future-based-s25-tok256-stage60m-20260309_152845_resume` |
| `6523165` | `default` | `online-cot-default-prompt-s25-tok256-stage60m-20260309_152845` | `1,802,240` steps (`Update 220/7324`) | `/data/group_data/rl/geney/online_rl_hidden_models/online-cot-default-prompt-s25-tok256-stage60m-20260309_152845_resume` |
| `6523695` | `future_based_opt` | `online-cot-more-future-based-opt-s25-tok256-stage60m-20260309_165005` | `1,720,320` steps (`Update 210/7324`) | `/data/group_data/rl/geney/online_rl_hidden_models/online-cot-more-future-based-opt-s25-tok256-stage60m-20260309_165005_resume` |

CoT log files:
- `/data/group_data/rl/geney/online_rl_hidden_models/cot_logs/online-cot-more-future-based-s25-tok256-stage60m-20260309_152845.jsonl`
- `/data/group_data/rl/geney/online_rl_hidden_models/cot_logs/online-cot-default-prompt-s25-tok256-stage60m-20260309_152845.jsonl`
- `/data/group_data/rl/geney/online_rl_hidden_models/cot_logs/online-cot-more-future-based-opt-s25-tok256-stage60m-20260309_165005.jsonl`

## Hidden-State Extraction Incident + Fix

Observed failure mode:
- Log flood of `WARNING: Failed to load hidden state, using zero vector`.
- This occurred in early `future_based_opt` run (`6523539`, now canceled).

Root cause:
- Some vLLM completion responses can omit `kv_transfer_params.hidden_states_path`.
- Hidden state file may still exist on disk (`<completion_id>-*.safetensors`).

Fix (already applied):
- Commit: `24b9ea5`
- File: `utils/llm_extractor.py`
- Behavior:
  - fallback lookup by completion id pattern in `hidden_states_path` directory,
  - retry waits for file visibility/readability before fallback,
  - warning counters rate-limited.

Verification state:
- Replacement run `6523695` started with same config and no hidden-state load warnings.

## Standard Ops Commands (Copy/Paste)

Always use `zsh -lic` wrappers (per `AGENTS.md`).

Queue status:
```bash
zsh -lic 'ssh babel "squeue -u geney -o \"%.18i %.36j %.10T %.10M %.9P %.30R\""'
```

Pull logs for a job:
```bash
zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && scripts/shell/babel.sh logs <jobid>'
```

Check hidden-state health:
```bash
zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && rg -n "Failed to load hidden state|hidden_states_path missing|Traceback|ERROR" logs/online_rl_hidden_jax_<jobid>.out'
```

## Relaunch Recipe If A CoT Run Fails

1. Pull logs and identify root cause.
2. If hidden-state load failures recur, verify extractor fix is present on Babel.
3. Cancel failing job:
```bash
zsh -lic 'ssh babel "scancel <jobid>"'
```
4. Submit identical replacement (example for `future_based_opt`, skip25, tok256, stage60m):
```bash
zsh -lic 'ssh babel "cd ~/Craftax_Baselines && TS=$(date +%Y%m%d_%H%M%S) && RUN_NAME=online-cot-more-future-based-opt-s25-tok256-stage60m-${TS} && CKPT_DIR=/data/group_data/rl/geney/online_rl_hidden_models/${RUN_NAME}_resume && COT_FILE=/data/group_data/rl/geney/online_rl_hidden_models/cot_logs/${RUN_NAME}.jsonl && sbatch --job-name online-cot-more-future-b --export=ALL,RUN_NAME=${RUN_NAME},CHECKPOINT_DIR=${CKPT_DIR},RESUME_FROM=,PROMPT_VARIANT=future_based_opt,HIDDEN_POOLING=last_token,HIDDEN_POOLING_K=8,TEMPERATURE=0.7,COT_LOG_TEXT=1,COT_LOG_FIRST_UPDATES=10,COT_LOG_EVERY_UPDATES=100,COT_LOG_SAMPLES=2,COT_LOG_MAX_CHARS=0,COT_LOG_FILE=${COT_FILE},CHAIN_ENABLED=1,CHAIN_TARGET_STEPS=60000000,CHAIN_CHECKPOINT_DIR=${CKPT_DIR},CHAIN_RUN_NAME=${RUN_NAME} scripts/sbatch/run_online_rl_hidden_jax.sbatch 128 60000000 25 24 256 64 1000000 /data/group_data/rl/geney/online_rl_hidden_models"'
```

## CoT Log Rendering To Markdown

Script:
- `scripts/render_cot_jsonl_to_markdown.py`

Usage on compute node (for `/data/...` source files):
```bash
zsh -lic 'ssh babel "srun --jobid=<running_jobid> --overlap --ntasks=1 --cpus-per-task=1 bash -lc '\''cd ~/Craftax_Baselines && python3 scripts/render_cot_jsonl_to_markdown.py --input /data/group_data/rl/geney/online_rl_hidden_models/cot_logs/<run>.jsonl --output /home/geney/Craftax_Baselines/logs/<run>.md'\''"'
```

## Expectations For Next Agents

- Keep explicit-path sync only (`scripts/shell/babel.sh push/pull <path...>`).
- Do not delete logs.
- Before new submissions, confirm:
  - no active hidden-state warning flood,
  - CoT hold guard is running for staged promotion waves,
  - chain jobs are checkpoint-progressing.
- When user reports an error, follow correction protocol in `AGENTS.md`:
  - read logs first,
  - add reusable rule,
  - then resubmit.
