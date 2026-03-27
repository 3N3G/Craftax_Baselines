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
  - After pushing edited benchmark/launcher scripts that will be submitted immediately, verify the specific remote file blocks you changed before resubmitting if the last run's behavior contradicted the local patch. Do not assume a successful `babel.sh push` log alone proves the target block landed on Babel.

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
  - When embedding inline Python in shell heredocs inside sbatch scripts, inspect the final script block after edits for accidental shell-level indentation or escaped Python quotes (`f\"...\"`); `bash -n` will not catch those Python syntax errors before submission.
  - Before submitting any job that includes inline Python patch blocks in shell heredocs, run a direct Python syntax preflight over those blocks (or at minimum grep for accidental escaped Python quotes like `f\"`) instead of relying on `bash -n` only. `6685007` failed before benchmark launch due an inline `f\"...\"` syntax error in a runtime patch block.
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
  - Qwen3.5 + vLLM 0.17.1 + `ExampleHiddenStatesConnector` plugin/server path is currently not stable here: both `Qwen3.5-9B` and `Qwen3.5-27B` reproduced a warmup-time CUDA index-out-of-bounds/device-side assert inside the connector (`slot_mapping` / KV-cache indexing). Do not keep resubmitting that plugin path blindly; switch Qwen3.5 hidden-state latency runs to a different extraction path (for example `vllm_offline`) or fall back to HF until the connector bug is fixed.
- Qwen3.5 hidden-state benchmark reliability rule:
  - For vLLM `0.17.x` + Qwen3.5 hybrid-cache models, the hidden-state connector path needs HMA-aware request-finished handling; patch `ExampleHiddenStatesConnector` to implement `SupportsHMA.request_finished_all_groups` (or patch scheduler fallback) before resubmitting.
  - For Qwen3.5 `vllm_offline` runs on vLLM `0.17.x` with `ExampleHiddenStatesConnector`, neither raw mode works by itself: HMA-on reproduced `ValueError: Connector ExampleHiddenStatesConnector does not support HMA but HMA is enabled` (`6670206`), while HMA-off reproduced `ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type` after model load/compile (`6671039`).
  - For the non-HMA Qwen3.5 `vllm_offline` path on vLLM `0.17.x`, patching only `unify_kv_cache_spec_page_size(...)` is not enough. `6682229` still failed after full model load inside `unify_hybrid_kv_cache_specs`, which raises before the later page-size helper is reached. Patch that disabled-HMA guard to call `unify_kv_cache_spec_page_size(...)` and update `kv_cache_spec` before raising.
  - `6682462` showed that even the page-size-unify fallback can still fail because Qwen3.5's disabled-HMA hybrid spec mix is not “one uniform type” after conversion. In that case, do not keep enforcing the final uniform-type check inside `unify_hybrid_kv_cache_specs`; let the function return and allow `get_kv_cache_groups(...)` to continue into its normal mixed-type page-size grouping path.
  - The current candidate path for `vllm_offline` is: keep the repo connector module `utils.vllm_hidden_connector` in `mode=last_token`, patch `v1/core/kv_cache_utils.py` page-size unification fallback, patch the scheduler request-finished compatibility path, patch/bypass the connector HMA support gate, and then run with HMA enabled.
  - Do not assume the repo connector module path alone makes vLLM treat the connector as HMA-capable; if the offline path still dies at the HMA support gate, inspect whether the runtime patch block actually ran for that backend.
  - For CacheOnlyAttention hidden-state connectors on vLLM `0.17.x`, do not reconstruct slot mappings from request block IDs when `attn_metadata.slot_mapping` is already available. `6671662` reached the first request and then died with a device-side index assert because the connector indexed KV pages using block-id-derived slot mappings instead of the live attention metadata mapping.
  - For Qwen3.5/vLLM hidden-state connector saves under HMA or cache-only attention, do not assume `request.token_ids.shape[0] == request.slot_mapping.shape[0]`; `6671662` showed prompt token ids can be longer than the live slot mapping. Slice saved `token_ids` and extraction length to the effective slot-mapping length to avoid mismatched hidden dumps or out-of-range cache indexing.
  - `6682601` showed that even after switching to `attn_metadata.slot_mapping`, some Qwen3.5 `vllm_offline` cache-only requests can still present invalid slot ids relative to the live `kv_layer.flatten(0, 1)` capacity. Filter slot mappings against the runtime KV capacity before indexing, keep `token_ids` aligned to the filtered positions, and honor `mode=last_token` in the patched `ExampleHiddenStatesConnector` instead of extracting the full prompt when only the last token is needed.
  - For single-model Qwen latency smoke/debug runs, do not rely on the sbatch/launcher filename to scope work. `6682769` still started a `Qwen3.5-27B` mirror download because `RUN_QWEN27B` defaults to `1`. When validating one targeted row, explicitly set every unrelated `RUN_QWEN*` and `RUN_QWEN*_HF` flag to `0` at submission time.
  - If the patched hidden connector overrides `clear_connector_metadata()`, do not leave it as a no-op. `6682819`/`6682977` isolated a mixed-prompt failure mode where `history_k5 -> state_only` dies in the second request while fresh `state_only` runners succeed; stale connector metadata is a plausible contributor and the override must reset `self._connector_metadata = None`.
  - Do not patch `clear_connector_metadata()` with an exact whole-block string match that hard-fails on minor upstream body changes. `6685025` failed before benchmark launch because the runtime patch expected exactly `pass`; patch the function structurally (by def signature + method block replacement) and warn instead of aborting if the method is absent.
  - For Qwen3.5 `vllm_offline` latency runs, do not require mixed prompt variants to share one engine instance. If long-history then short-state requests are unstable in one runner, benchmark `predict_history_k5` and `predict_state_only` in separate fresh runners and merge the resulting `records.jsonl`/`summary.json` into the canonical run label.
  - For Qwen3.5 `vllm_offline` runs on vLLM `0.17.x`, once the KV page-size fallback patch is in place, prefer serving the local compat model dir with HMA disabled. `6681411` showed the HMA-enabled docs/connector path can get past startup/model load and still fail in `save_kv_layer` with malformed slot mappings under hybrid cache groups.
  - Do not count a Qwen3.5 hidden-state call as valid just because a `.safetensors` file loads. Sanity-check the hidden tensor shape against the expected extracted-layer count; `6681411` produced an apparent success with shape `[1056, 1, 4096]`, which indicates a malformed layout for a single-layer extraction path.
  - Qwen3.5 HF baselines must run with a transformers build that recognizes `model_type=qwen3_5` (source `transformers` overlay or equivalent); stock older releases in stable envs are not sufficient.
  - Do not prepend the Qwen3.5 HF/source-transformers overlay to the vLLM server `PYTHONPATH`; keep that overlay HF-only, or vLLM can fail during config validation inside `qwen3_5.py`/`huggingface_hub` before benchmark requests start.
  - For reusable latency-benchmark overlays on Babel, do not assume `/data/user_data/$USER/...` is writable from compute/login shells; use a known writable shared path under `/home/$USER` if you want overlay reuse across jobs.
- Latency benchmarking semantics rule:
  - Do not compare online-RL hidden-state extraction latency against future-prediction generation latency as if they were the same workload.
  - For latency reports, explicitly separate: model/server startup, warm-path hidden-only extraction, and generation latency with stated `max_tokens`.
  - Every benchmark table must state prompt token counts, completion token counts, whether warmup was used, and whether the server/model was already resident.
- CoT submission reliability rule:
  - For multi-job CoT waves, launch/verify `scripts/shell/submit_cot_hold_guard.sh` so pending `cot_*` jobs that enter `JobHeldUser` are auto-released during unattended runs.
- Text-observation coordinate integrity rule:
  - Never parse `Map:` entries with naive comma splitting; split entries by coordinate anchors (`-?\d+,\s*-?\d+\s*:`) so row/col pairs cannot be separated.
  - Any emitted `Map (interesting tiles only)` line must validate as repeated `row,col:tile` tokens; malformed tokens like `-5:tree` are fatal in online RL prompt paths.
  - If filtering/parsing fails in non-training tools, preserve original map text rather than emitting partially corrupted coordinates.
- RL node env-path reliability rule:
  - Before launching `run_online_rl_hidden_jax.sbatch`, ensure `TRAIN_ENV_PATH`/`VLLM_ENV_PATH` exist on the target node family; some RL nodes can miss `/data/user_data/geney/.conda/envs/*` even when others have it.
  - If a segment fails at `training env import preflight`, rerun on a node family with known-good env mounts (or set explicit existing env paths) before resubmitting the chain.
  - If a vLLM benchmark run reaches engine init/model-load logs and then stops emitting output for multiple minutes while GPU memory is allocated but utilization stays ~0% (for example `6685045` on `babel-n9-20`), treat it as a node-local stall: cancel and resubmit excluding that node (or to a known-good node family) instead of waiting indefinitely.
- Latency benchmark fairness rule:
  - Do not compare vLLM and HF latency from mismatched execution paths (for example native/offline vLLM vs API/plugin vLLM, or hidden-only vs generation-heavy runs) and then draw speed conclusions from that table.
  - For hidden-state latency comparisons, record the exact backend (`vllm_offline`, API/plugin, or HF), prompt variant, benchmark mode, and whether `enforce_eager`, prefix caching, or chunked prefill were enabled; keep those settings explicit in the report.
- vLLM/HF env split rule:
  - Do not upgrade a shared vLLM runtime overlay to a bleeding-edge `transformers`/`huggingface_hub` build just to enable newer HF model families; this can break vLLM config validation on the same job.
  - When HF requires newer `transformers` than vLLM supports, keep a stable vLLM env for native benchmarks and build a separate HF-only env for the source-`transformers` rows.
  - When the base env already contains the heavy CUDA/PyTorch stack, `python -m venv --system-site-packages` is only safe if that base env does not also carry incompatible inference/runtime packages. If the base env already has packages like `sglang`, `triton_kernels`, or another `vllm`, prefer an isolated job-local venv instead of inheriting site-packages; mixed imports can survive overlay upgrades and produce misleading Triton/kernel failures.
  - For heavy native-vLLM overlays, also preflight whether node-local `/tmp`/`${SLURM_TMPDIR}` can actually hold a full `torch`+`vllm` reinstall. `6670982` failed with `Errno 28 No space left on device` during the isolated overlay install; in that case, fall back to a `--system-site-packages` overlay and upgrade only the packages under test instead of insisting on a clean reinstall that cannot fit.
- Qwen3.5 config-compatibility rule:
  - Do not rely on `transformers.AutoConfig.from_pretrained(...)` to read Qwen3.5 metadata in utility/benchmark code; some installed transformer builds still do not register `model_type=qwen3_5` even when the model otherwise loads with `trust_remote_code`.
  - For Qwen3.5 layer-count or architecture inspection, parse raw `config.json` (or alias-register `qwen3_5`) before resubmitting jobs.
  - For native vLLM Qwen3.5 hidden-state runs, patch `vllm.transformers_utils.config.get_hf_text_config` to wrap dict-valued nested `text_config` objects before benchmarking; otherwise speculative hidden-state init can fail even when `num_attention_heads` exists in the raw JSON.
  - When preparing Qwen3.5 compat model dirs for vLLM, set the architecture explicitly to the causal loader (`Qwen3_5ForCausalLM` or `Qwen3_5MoeForCausalLM`) instead of inheriting `*ForConditionalGeneration`.
  - For native vLLM Qwen3.5 hidden-state runs, patch `vllm/v1/worker/gpu_model_runner.py` to tolerate aux-hidden-state model forwards that return more than two values; qwen3.5 can reach profile-run failure with `ValueError: too many values to unpack (expected 2)` unless the runner takes the first two outputs and ignores extras.
- vLLM patch-write reliability rule:
  - Do not patch `vllm` or hidden-connector files in-place under shared primary conda env site-packages during benchmark/eval runs; node/user quotas can fail these writes.
  - Before building a large reusable vLLM overlay/venv on shared storage (`/home`, `/data/user_data`, etc.), preflight actual writable quota/headroom; do not trust `df` alone. If shared quota is tight, use node-local `/scratch`/`${SLURM_TMPDIR}` for one-shot job-local overlays instead of failing mid-install with `Errno 122 Disk quota exceeded`.
  - When building a job-local overlay on `/scratch` or `${SLURM_TMPDIR}`, also force installer/cache/temp roots (`PIP_CACHE_DIR`, `XDG_CACHE_HOME`, `TMPDIR`, and similar) into that same job-local root, or disable pip caching with `--no-cache-dir`; otherwise `pip` may still write large artifacts under `~/.cache` and fail on home-directory quota even though the venv itself is on scratch.
  - When shadow-patching vendored `vllm` source files, avoid exact whole-block string matches when a smaller structural patch will do; minor upstream formatting drift can otherwise abort the job before benchmarking starts. Prefer a narrow replacement around the failing guard/statement and only hard-fail after checking the target version's actual file contents.
  - For qwen3.5 `vllm_offline` hidden-state runs on vLLM >= 0.17 with `ExampleHiddenStatesConnector`, the non-HMA fallback is not sufficient by itself: `6671039` still failed after model load with `Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type`. If you need the offline path to work, make sure the HMA gate/scheduler compatibility patches are applied to that backend and retry with HMA enabled.
  - If a qwen3.5 `vllm_offline` non-HMA run gets through full weight load/compile and then dies in `unify_hybrid_kv_cache_specs` (`6682229`), do not assume the earlier page-size patch applied to the relevant branch. Inspect the runtime patch block and patch the disabled-HMA `unify_hybrid_kv_cache_specs` guard itself, not just `unify_kv_cache_spec_page_size`.
  - Default to shadow-copy patching under `${TMP_ROOT}` and prepend that path via `PYTHONPATH`; only patch primary env in-place when explicitly requested.
  - For qwen3.5 compatibility runs where a local patched compat model is prepared, serve vLLM from that local compat directory (with injected layer IDs) instead of remote model IDs/config wrappers, to avoid startup stalls on remote model resolution.
  - If vLLM startup sits in `rpc_wait_bit_killable` with zero GPU use, move `TMP_ROOT`/artifact paths to node-local storage (`${SLURM_TMPDIR}` or `/tmp`) rather than shared `/scratch` mounts, then rerun.
  - When forcing qwen3.5 weights through qwen3 loaders, normalize checkpoint keys for `model.language_model.*` (and related wrapper variants) to `model.*`; otherwise vLLM will fail engine init with many “weights were not initialized from checkpoint” errors.
  - Do not assume the hybrid-KV-cache-manager flag is universally correct for qwen3.5 hidden-state connector runs on vLLM >=0.17; for the current `vllm_offline` + `ExampleHiddenStatesConnector` path, HMA-off and HMA-on fail differently unless the offline runtime patch block applies the connector-compatibility patches too.
  - When layering `vllm-hidden-states-extractor` onto a newer vLLM overlay (for example `vllm==0.17.x`), install the plugin with `pip install --no-deps ...`; its package metadata pins `vllm==0.14.0`, and allowing pip to resolve deps will break the overlay before the benchmark starts.
  - For API/plugin hidden-state benchmarks that rely on `vllm-hidden-states-extractor`, prefer the known-good `vllm==0.14.0` path unless the config generator has been updated for newer `extract_hidden_states` draft-model requirements; on `vllm>=0.17`, Qwen3 server startup can fail with `eagle_aux_hidden_state_layer_ids must be set in the draft model config`.
- HF download/cache reliability rule:
  - For HF-based eval/benchmark jobs, force `HF_HOME`/`HUGGINGFACE_HUB_CACHE`/`TRANSFORMERS_CACHE` into the per-job temp root (`${SLURM_TMPDIR}` or explicit `TMP_ROOT`) instead of inheriting shell defaults under `/data/user_data/$USER`, to avoid quota-triggered `Errno 122`.
  - For large-model snapshot downloads that fail with Xet reconstruction/writer errors, disable Xet transport (`HF_HUB_DISABLE_XET=1`) before `snapshot_download`/`from_pretrained` retries.
  - When enabling temporary transformers overlays, do not couple readiness/preflight to immediate remote model downloads; keep preflight to local import/version checks so early failures are environment issues, not transient network/cache errors.
  - For repeated Babel vLLM/HF benchmarks on public Hugging Face model IDs, do not keep launching jobs against raw HF ids after `429 Too Many Requests` / `LocalEntryNotFoundError` startup failures. First materialize a reusable local model mirror/cache path (authenticated HF or alternate mirror such as ModelScope), then point the benchmark at that local path so engine startup no longer depends on live Hub metadata calls.
