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
- Keep diagnostics: do not delete `~/Craftax_Baselines/logs` contents unless explicitly requested.

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
