# Babel Ops Reference (zsh -lic First)

Use this as a minimal, safe reference for local<->Babel development.

## 1) Canonical Command Pattern
- Run remote commands through:
  - `zsh -lic 'ssh babel "<remote-cmd>"'`
- Run local helper commands through:
  - `zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && <local-cmd>'`

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
- Never assume env names are valid on every node/session; use absolute env paths first, then fallbacks.
- For eval/training entrypoints, verify env correctness with a python import preflight for required modules before launching the main command.
- Preflight must include workflow-specific runtime deps (for policy eval this includes `distrax` and model-import deps), not just generic libs.
- Run eval-env preflight before expensive service startup (like vLLM) so bad envs fail fast without burning GPU startup time.
- Prefer a single known-good env for both vLLM + eval when possible (to avoid CUDA/torch ABI mismatch across envs on RL nodes).
- If multiple envs are candidates (`craftax_fast_llm`, `test`, etc.), pick the one proven stable on the target partition/node type, not just any node.
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
- When loading `ActorCriticAug` checkpoints, infer `action_dim` from the final action head (`actor_out`) first; never infer it from intermediate actor MLP layers when `actor_out` exists.
- Training/eval consistency rule for hidden-state policies:
  - Always evaluate a checkpoint with the same policy family and architecture it was trained with (e.g. `ActorCriticAug` vs JAX actor-critic, fusion mode, hidden dim, layer width, actor/critic head depth).
  - Keep hidden-state construction identical to training: same text preprocessing/prompt path (including `filter_text_obs` usage and template), same hidden extraction mode, and same normalization stats (`hidden_state_stats.npz`) paired to that checkpoint.
  - Keep LLM layer consistent with training data provenance: do not change `llm_layer` in manifests as a shortcut; if training used layer `-1`, eval must use `-1` unless a new model is retrained on layer `24` data.
  - Keep hidden refresh cadence semantics consistent (`skip_n`, reset-on-done behavior) when comparing policies.
  - Before submitting eval waves, explicitly verify each policy entry maps to the correct checkpoint + stats + layer tuple from the corresponding training run metadata.
