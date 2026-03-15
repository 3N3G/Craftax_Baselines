# Babel Ops Reference (zsh -lic First)

Use this as a minimal, safe reference for local<->Babel development.

## 0) Codex Continuity
- For active experiment status/handoff context before submitting new jobs, read:
  - `docs/CODEX_AGENT_HANDOFF.md`

## 1) Canonical Command Pattern
- Run remote commands through:
  - `zsh -lic 'ssh babel "<remote-cmd>"'`
- Run local helper commands through:
  - `zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && <local-cmd>'`
- Execution location rule:
  - Local Craftax (for example via `imaug`) is inspection-only (package/source checks, import/introspection, static verification).
  - All actual code execution for this repo (evals, training, rollouts, sbatch entrypoints, python scripts that run environments) must be run on Babel.
  - If uncertain, default to Babel.

## 2) Sync Rules (Explicit Paths Only)
- Use targeted sync via helper:
  - Local -> Babel:
    - `zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && scripts/shell/babel.sh push <path1> <path2>'`
  - Babel -> Local:
    - `zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && scripts/shell/babel.sh pull <path1> <path2>'`
- Pull logs for a job:
  - `zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && scripts/shell/babel.sh logs <jobid>'`

## 3) Hard Safety Constraints
- Do not use broad repo sync commands in normal workflow.
- Do not use `push --all --delete`.
- Do not use raw `rsync` against `~/Craftax_Baselines`.
- Keep diagnostics: do not delete `~/Craftax_Baselines/logs` contents EVER unless explicitly requested.
- Do not run Craftax experiments locally; local machine usage is limited to inspection and file editing.

## 4) Slurm Basics
- Queue view:
  - `zsh -lic 'ssh babel "squeue -u geney -o \"%.18i %.24j %.10T %.10M %.9P %.30R\""'`
- Submit:
  - `zsh -lic 'ssh babel "cd ~/Craftax_Baselines && sbatch <script> <args...>"'`
- Accounting:
  - `zsh -lic 'ssh babel "sacct -j <jobid> --format=JobID,State,ExitCode,Reason,Elapsed,NodeList -P -n"'`

## 5) Runtime/Logs Notes
- Default Slurm logs:
  - `~/Craftax_Baselines/logs/*.out`, `~/Craftax_Baselines/logs/*.err`
- Online RL also mirrors runtime logs to:
  - `/data/user_data/$USER/craftax_job_logs`
- If expected job logs are missing, check:
  - `sacct` state/exit first
  - `scripts/shell/babel.sh logs <jobid>` for both repo and mirrored logs

## 6) Sbatch Authoring Guardrails
- Before writing or changing an sbatch script, inspect at least 2 existing working sbatch scripts for the same workflow family and copy their env/bootstrap patterns.
- Never assume env names are valid on every node/session; pin one explicit absolute env path per workflow and fail fast if it cannot be activated.
- For eval/training entrypoints, verify env correctness with a python import preflight for required modules before launching the main command.
- Preflight must include workflow-specific runtime deps (for policy eval this includes `distrax` and model-import deps), not just generic libs.
- Run eval-env preflight before expensive service startup (like vLLM) so bad envs fail fast without burning GPU startup time.
- Prefer a single known-good env for both vLLM + eval when possible (to avoid CUDA/torch ABI mismatch across envs on RL nodes).
- If multiple envs are candidates (`craftax_fast_llm`, `test`, etc.), pick the one proven stable on the target partition/node type, not just any node.
- In sbatch scripts, avoid candidate env lists and `conda activate ... || ...` chains; expose a single `<WORKFLOW>_ENV_PATH` override variable with a deterministic default.
- Default W&B routing:
  - eval jobs -> `craftax_symbolic_evals`
  - unaugmented symbolic PPO training -> `unaugmented_craftax_ppo`
  - only override these defaults when the user explicitly requests a different project.
- For cross-node eval jobs, avoid defaulting to envs with known node-specific CUDA ABI issues (`test`); prefer the stable eval env (`imaug`) unless logs prove otherwise.
- Do not default heavy vLLM temp artifacts to `/tmp`; prefer `${SLURM_TMPDIR}`/`/scratch` and only fall back to `/tmp` if creation succeeds.
- If vLLM jobs can run in parallel, isolate each job with:
  - unique `VLLM_URL`/port
  - unique `VLLM_TMP_ROOT`
  - port spacing large enough for vLLM internal auxiliary port probing (not just one distinct API port).

## 7) Correction Protocol
- When user points out a mistake, do this before further submissions:
  - read the actual error logs and identify root cause
  - add a reusable prevention rule to this file (`AGENTS.md`)
  - apply the fix and only then resubmit jobs.
- Treat each correction as a general principle, not a one-off patch.
- Remote shell quoting reliability rule:
  - For complex remote loops/heredocs (especially with `*`, `$()`, or jq filters), run via `ssh babel "bash -s"` with a single-quoted heredoc (`<<'EOF' ... EOF`) to avoid local zsh globbing/substitution.
  - Inside double-quoted remote strings, escape remote-only variables as `\$var`/`\${var}` so they are expanded on Babel, not locally.
  - Do not rely on unescaped globs in outer `zsh -lic` strings; if command complexity grows, move logic into a checked script and call it remotely.
- RL QoS sync reliability rule:
  - After editing any local sbatch header affecting `--partition`/`--qos`, push those exact scripts to Babel with explicit-path `scripts/shell/babel.sh push <path...>` before submitting jobs.
  - Before submission/resubmission, verify the remote script header includes `#SBATCH --partition=rl` and `#SBATCH --qos=rl_qos` (for RL jobs).
  - If a pending RL job is blocked with `QOS=normal` and `Job's QOS not permitted`, update it in place (`scontrol update JobId=<id> QOS=rl_qos`) and re-check `squeue`.
- RL checkpoint quota reliability rule:
  - Before launching/resuming `run_online_rl_hidden_jax.sbatch`, verify the selected `POLICY_SAVE_DIR`/`CHECKPOINT_DIR` are writable on the target compute node family; fail fast before vLLM startup if write-probe fails.
  - If `/data/group_data/rl` is at quota/100%, route new checkpoints and CoT logs to a writable user path (for example `/data/user_data/$USER/...`) and keep `RESUME_FROM` pointed at the latest readable checkpoint source.
  - A recurring `Errno 122 Disk quota exceeded` at policy snapshot time indicates progress will loop between old checkpoints unless checkpoint output path is moved.
- Large archive reliability rule:
  - Do not assume a single-node `/scratch` can hold full-dataset tarballs; preflight writable free space on the target node family before starting long archive jobs.
  - For multi-terabyte directories under quota pressure, prefer resumable sharded archives with durable temp staging (for example `/data/user_data/$USER/...`) over one giant `/scratch/*.tar.gz`.
  - Only delete source chunks after the corresponding shard passes integrity validation and is durably moved to destination storage.
- Long-run PPO checkpoint reliability rule:
  - Do not invoke `jax.experimental.io_callback` with full policy params every update just to test save cadence; gate callback execution so params are transferred only when a save is actually due.
  - Periodic checkpoint triggers must use interval-crossing semantics (prev_step//N < curr_step//N), not exact modulo equality, because PPO update stride (`NUM_STEPS*NUM_ENVS`) often does not divide desired save intervals.
  - For very long continuation runs, default periodic policy saves to off (`SAVE_POLICY_EVERY_STEPS=0`) unless intermediate checkpoints are explicitly needed.
- When loading `ActorCriticAug` checkpoints, infer `action_dim` from the final action head (`actor_out`) first; never infer it from intermediate actor MLP layers when `actor_out` exists.
- Training/eval consistency rule for hidden-state policies:
  - Always evaluate a checkpoint with the same policy family and architecture it was trained with (e.g. `ActorCriticAug` vs JAX actor-critic, fusion mode, hidden dim, layer width, actor/critic head depth).
  - Keep hidden-state construction identical to training: same text preprocessing/prompt path (including `filter_text_obs` usage and template), same hidden extraction mode, and same normalization stats (`hidden_state_stats.npz`) paired to that checkpoint.
  - Keep LLM layer consistent with training data provenance: do not change `llm_layer` in manifests as a shortcut; if training used layer `-1`, eval must use `-1` unless a new model is retrained on layer `24` data.
  - Keep hidden refresh cadence semantics consistent (`skip_n`, reset-on-done behavior) when comparing policies.
  - Before submitting eval waves, explicitly verify each policy entry maps to the correct checkpoint + stats + layer tuple from the corresponding training run metadata.
- vLLM hidden-state extraction reliability rule:
  - Do not assume `/v1/completions` always returns `kv_transfer_params.hidden_states_path`; some responses can omit it while still writing the `.safetensors` file.
  - Extractors must include a completion-id (`result["id"]`) filename fallback and short file-availability retry before zero-vector fallback.
  - If `Failed to load hidden state` appears, inspect the actual JSON response shape first and verify hidden-state file creation on the compute node before changing prompts.
- CoT submission reliability rule:
  - For multi-job CoT waves, launch/verify `scripts/shell/submit_cot_hold_guard.sh` so pending `cot_*` jobs that enter `JobHeldUser` are auto-released during unattended runs.
- Text-observation coordinate integrity rule:
  - Never parse `Map:` entries with naive comma splitting; split entries by coordinate anchors (`-?\d+,\s*-?\d+\s*:`) so row/col pairs cannot be separated.
  - Any emitted `Map (interesting tiles only)` line must validate as repeated `row,col:tile` tokens; malformed tokens like `-5:tree` are fatal in online RL prompt paths.
  - If filtering/parsing fails in non-training tools, preserve original map text rather than emitting partially corrupted coordinates.
- RL node env-path reliability rule:
  - Before launching `run_online_rl_hidden_jax.sbatch`, ensure `TRAIN_ENV_PATH`/`VLLM_ENV_PATH` exist on the target node family; some RL nodes can miss `/data/user_data/geney/.conda/envs/*` even when others have it.
  - If a segment fails at `training env import preflight`, rerun on a node family with known-good env mounts (or set explicit existing env paths) before resubmitting the chain.
