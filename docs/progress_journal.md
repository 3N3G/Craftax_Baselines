# Craftax LLM-Augmented RL — Progress Journal

Running log of experiments, results, and next steps.

---

## 2026-02-19: JAX Online RL Port Completed

### Summary
Ported `online_rl_hidden.py` (PyTorch) → `online_rl_hidden_jax.py` (pure JAX/Flax) to eliminate
framework-crossing overhead (JAX→numpy→torch→numpy→JAX per step).

### Performance Results

| Configuration | SPS | Notes |
|---|---|---|
| PyTorch version (`online_rl_hidden.py`) | ~150 | Framework conversions every step |
| **JAX version (`online_rl_hidden_jax.py`)** | **~750** | Pure JAX, JIT-compiled policy |
| Baseline PPO (no LLM) | ~18,500 | Reference upper bound |

**Config**: 128 envs, skip_n=100000000 (effectively ∞, LLM called only on step 1)

### Analysis: Is 750 SPS optimal?

With `skip_n=∞`, only the first step invokes the LLM. Subsequent steps are purely:
1. **JIT policy forward pass** — should be sub-ms after warmup
2. **Vectorized env step** via `jax.vmap(env.step)` — should be fast

The gap from 750 → 18,500 SPS is likely due to:
- **Text rendering** on step 1 still iterates 128 envs in a Python loop (`render_craftax_text_swapped` + `filter_text_obs`)
- **JIT compilation overhead** — first few steps compile the policy and env step functions, dragging down the average
- **Non-JIT'd Python overhead** in the training loop (metrics computation, numpy conversions for action tracking)
- The env+policy are on **CPU** (forced by `JAX_PLATFORM_NAME=cpu` to avoid GPU conflicts with vLLM)

**Key insight**: Baseline PPO runs JAX on GPU with everything JIT'd inside a `jax.lax.scan`. Our loop is a Python for-loop on CPU. The 25x gap is expected for CPU-based Python-loop JAX vs GPU-scanned JAX.

### Bugs Fixed in JAX Port
1. `hidden_state_dim` hardcoded `Config.HIDDEN_SIZE` → dynamic `self.llm.hidden_size`
2. Extra `batch_size` arg on `extract_hidden_states_no_cot` call
3. Missing per-episode reward tracking (no `episode_reward_mean` in WandB)
4. Silent `try/except` on achievement tracking
5. Achievement state not reset on episode done
6. Missing `TOKENIZERS_PARALLELISM` env var
7. Deprecated `jax.tree_map` → `jax.tree.map` (JAX v0.6.0)

### Conclusion
**750 SPS is approximately optimal for CPU-based JAX with a Python training loop.** The real speedup opportunity would be running the env+policy on GPU (requires separate GPU from vLLM or time-sharing), or using `jax.lax.scan` to JIT the entire rollout. Neither is necessary right now since the focus is shifting to offline training.

---

## Current Focus: Offline Training with LLM-Augmented AWR

The online RL infrastructure is now functional. Focus shifts to offline RL experiments using pre-labelled datasets with LLM hidden states.

### Infrastructure Status

| Component | Status | Notes |
|---|---|---|
| Data generation (`online_rl/ppo.py`) | ✅ Ready | Saves symbolic trajectories |
| Redis job queue | ✅ Ready | `labelling/addtoqueue_llm.py` |
| LLM labelling workers | ✅ Ready | `labelling/llm_worker.py`, `makeworkers_llm.sbatch` |
| vLLM hidden state extractor | ✅ Ready | `utils/llm_extractor.py` |
| Offline AWR baseline | ✅ Ready | `offline_rl/awr.py` |
| Offline AWR augmented | ✅ Ready | `offline_rl/awr_llm_augmented.py` |
| Eval baseline | ✅ Ready | `offline_rl/eval_awr.py` |
| Eval augmented | ⚠️ Uses old VLM server | `offline_rl/eval_awr_vlm_augmented.py` — needs update to vLLM |

### Data Paths
- Unlabelled: `/data/group_data/rl/geney/craftax_unlabelled_symbolic/`
- Labelled: `/data/group_data/rl/geney/craftax_labelled_results/`
- With returns: `/data/group_data/rl/geney/craftax_labelled_results_with_returns/`

---

## 2026-02-21: Offline AWR Startup Hardening

### What Changed
- Updated `offline_rl/awr_llm_augmented.py` to support live, partially-complete labelled datasets:
  - Default data path now points to actively-produced vLLM labels: `/data/group_data/rl/geney/vllm_craftax_labelled_results`
  - If `return_to_go` is missing, it is computed on load from `reward`/`done` (interleaved multi-env aware, with fallback path)
  - Added partial-run controls for fast iteration:
    - `--max_files`
    - `--data_glob`
    - `--dataset_workers`
    - `--num_envs`
    - `--compute-missing-returns` / `--require-returns`
- Added robust sbatch launcher: `scripts/sbatch/run_offline_awr_llm_augmented.sbatch`
  - Handles env bootstrap, local WandB dirs, and configurable run params via env vars.

### Why
- Offline experiments can now start before full dataset labelling completion.
- Removes the blocking dependency on pre-materializing a separate “with_returns” dataset for every new file.

### Validation
- Synthetic smoke run on cluster completed successfully via:
  - `scripts/sbatch/run_offline_awr_llm_augmented.sbatch`
  - `DATA_DIR=/home/geney/Craftax_Baselines/tmp_offline_smoke_data`
  - `TOTAL_STEPS=3`, `MAX_FILES=1`, `NO_WANDB=1`
  - Produced checkpoint: `/home/geney/Craftax_Baselines/tmp_offline_smoke_ckpt/awr_llm_final.pth`

---

## 2026-02-21: Offline Pair Runs + Value-Probing Utilities

### What Changed
- Updated `scripts/sbatch/run_offline_awr_llm_augmented.sbatch`:
  - Adds `HIDDEN_MODE` passthrough (`real|zero|shuffle`) so ablations can be launched from sbatch.
  - Keeps WandB default-on (only disabled if `NO_WANDB=1`).
  - Supports explicit `WANDB_NAME` passthrough for clean run grouping.
- Added paired launcher: `scripts/sbatch/launch_offline_awr_pair.sh`
  - Submits two comparable jobs automatically:
    - `HIDDEN_MODE=real`
    - `HIDDEN_MODE=zero`
  - Uses shared data/hparams and separates checkpoint dirs by run tag.
- Added value-function probe scripts:
  - `offline_rl/analyze_value_pairs.py`
    - Loads one or more AWR-LLM checkpoints.
    - Generates controlled state pairs from true Craftax `EnvState`.
    - Reports value deltas (`high - low`) and sign-consistency stats.
  - `offline_rl/generate_value_probe_pairs.py`
    - Generates/saves paired `EnvState` objects + symbolic vectors (`obs_low`, `obs_high`) for offline analysis.
    - Optional text render dump for inspection.

### Runtime Notes
- Fixed a cluster-only runtime bug in `offline_rl/awr_llm_augmented.py` (`Tuple` import missing under Python 3.10).
- Added memory-safety controls for offline loading:
  - `--max_dataset_gb` (default `80`)
  - auto file limiting (on by default; disable via `--disable_auto_file_limit`)
- This prevents silent OOM when labelled file count grows.

### Current Running Pair (stable)
- Active paired submission:
  - `6420313` (`awr_real`)
  - `6420314` (`awr_zero`)
- Config highlights:
  - WandB enabled
  - `MAX_DATASET_GB=24` (auto-limited to 72 files on this run)
  - `DATASET_WORKERS=2`
- Both jobs passed dataset load and reached training steps (no OOM).

### Pair Run Outcome
- `6420313` (`real`) and `6420314` (`zero`) completed successfully to 100k steps.
- Final checkpoints:
  - `/data/group_data/rl/geney/checkpoints/awr_llm_augmented_live/offline_pair_20260221_191229_real/awr_llm_final.pth`
  - `/data/group_data/rl/geney/checkpoints/awr_llm_augmented_live/offline_pair_20260221_191229_zero/awr_llm_final.pth`

### Probe Comparison Snapshot
- 25k checkpoint comparison (96 sampled states): `logs/value_pairs_25k_6420321.json`
- Final checkpoint comparison (64 sampled states): `logs/value_pairs_final_6420329.json`
- Note: these probes currently use `hidden_input_mode=zero`, so they isolate observation-branch value behavior; a follow-up pass should feed real hidden vectors from matched labelled samples.

---

## 2026-02-21: Exact RTG + Real-Hidden Pair Probe

### Exact RTG (cross-file, no boundary truncation)
- Added `tools/ppo_add_returns_exact.py`:
  - Computes return-to-go in reverse over the entire ordered file sequence.
  - Carries per-env return vectors across file boundaries (exact for PPO-saved interleaved streams).
  - Supports in-place overwrite or writing to a new output directory.
- Dry-run validation (`6420933`) on current labelled set:
  - Files: 589
  - Samples: 4,825,088
  - Confirmed expected cross-file stream carry behavior.
- Write job currently running:
  - Job: `6420992`
  - Target: `/data/group_data/rl/geney/vllm_craftax_labelled_results_with_returns_exact`

### Value Probe With Real LLM Hidden States
- Updated `offline_rl/analyze_value_pairs.py` with `--hidden_input_mode llm_no_cot`:
  - Uses `LLMHiddenStateExtractor.extract_hidden_states_no_cot(...)` to compute hidden vectors for each low/high counterfactual state.
- Updated `utils/llm_extractor.py` no-CoT path to run the backbone model directly (skip CausalLM logits allocation), which reduces GPU memory usage and avoids probe-time OOM in this workflow.
- Completed real-hidden probe run (`6420957`):
  - Output: `logs/value_pairs_realhidden_6420957.json`
  - Config: 6 base states, 6 pair types, CUDA, batch size 1 for LLM hidden extraction.
- Completed diversified real-hidden probe (`6421089`):
  - Output: `logs/value_pairs_realhidden_6421089.json`
  - Config: 8 base states sampled across 2400 rollout steps, 6 pair types, CUDA, batch size 1.
- Headline result:
  - Sensitivity ranking shifts when using real LLM hidden states; in the diversified 8-state run, the `zero` checkpoint shows stronger monotonicity than the `real` checkpoint on several scalar resource pairs.
  - This indicates the counterfactual probe is now exposing non-trivial hidden-state interactions rather than a simple always-positive trend.

### Early Probe Smoke Result (sanity check)
- `offline_rl/analyze_value_pairs.py` run on smoke checkpoint (`/data/group_data/rl/geney/checkpoints/awr_llm_augmented_smoke/awr_llm_final.pth`) completed.
- With `hidden_input_mode=zero` (32 states):
  - `health`: sign-correct 1.00
  - `food`: sign-correct 1.00
  - `drink`: sign-correct 1.00
  - `energy`: sign-correct 0.81
  - `wood`: sign-correct 1.00
  - `stone`: sign-correct 0.00 (unexpected direction; worth deeper check on full checkpoints)

---

## 2026-02-22: Labelling Throughput + Metadata Load Reduction (Next Rerun)

### What Changed
- Updated `labelling/llm_worker.py`:
  - Default WandB mode is now **worker-level** (one run per worker) instead of one run per file.
  - Per-file WandB remains available via `WANDB_PER_FILE=1`.
  - Reduced console log frequency with `PROGRESS_LOG_EVERY_BATCHES` (default 16).
  - Stage input with `shutil.copyfile(...)` to local scratch and keep temp/progress/WandB artifacts on local scratch.
  - Increment Redis completed counter (`craftax_llm_completed_count`) after each successful output write.
- Updated `labelling/run_labelling.sbatch`:
  - Initializes and reads Redis completed counter for monitoring progress.
  - Falls back to filesystem count only when counter is missing, then re-seeds Redis.
  - Keeps existing stalled-queue detection and worker relaunch behavior.
- Updated `labelling/makeworkers_llm.sbatch`:
  - Exposes defaults for `PROGRESS_LOG_EVERY_BATCHES` and `WANDB_PER_FILE`.

### Why
- Per-file WandB startup/teardown and per-batch logging were adding avoidable overhead.
- Redis-based progress tracking reduces repeated directory scans during monitoring.
- Local scratch staging continues to minimize NFS metadata pressure.

### Validation
- Syntax checks passed:
  - `python3 -m py_compile labelling/llm_worker.py`
  - `bash -n labelling/run_labelling.sbatch`
  - `bash -n labelling/makeworkers_llm.sbatch`
- Updated files synced to Babel for the next labelling rerun.

---

## 2026-02-22: Offline RL + Value Diagnostics (Exact RTG)

### Exact RTG Dataset
- `rtg_exact_write` job `6420992` completed successfully.
- Output dir: `/data/group_data/rl/geney/vllm_craftax_labelled_results_with_returns_exact`
- Summary from log:
  - Processed files: `590`
  - Processed samples: `4,833,280`
  - Runtime: `5489.9s`

### Offline AWR Pair Run (Exact RTG)
- Submitted with:
  - `DATA_DIR=/data/group_data/rl/geney/vllm_craftax_labelled_results_with_returns_exact`
  - `REQUIRE_RETURNS=1`
  - `MAX_DATASET_GB=24`
  - `DATASET_WORKERS=2`
  - `TOTAL_STEPS=100000`
- Jobs:
  - `6421528` (`awr_real`) COMPLETED
  - `6421529` (`awr_zero`) COMPLETED
- Both runs loaded exact-RTG files and reported:
  - `Computed missing returns for 0 files`
  - auto-limited to `72/590` files (memory budget)
- Final checkpoints:
  - `/data/group_data/rl/geney/checkpoints/awr_llm_augmented_live/offline_pair_exactrtg_20260222_011024_real/awr_llm_final.pth`
  - `/data/group_data/rl/geney/checkpoints/awr_llm_augmented_live/offline_pair_exactrtg_20260222_011024_zero/awr_llm_final.pth`

### Value-Learning Diagnostic Script Updates
- Added `offline_rl/analyze_value_learning.py`:
  - Evaluates value predictions vs RTG on sampled dataset transitions.
  - Metrics: Pearson/Spearman, EV, MSE/MAE, bias, top-minus-bottom RTG, calibration bins.
  - Supports Torch offline checkpoints and JAX online checkpoints.
- Updated `offline_rl/analyze_value_pairs.py`:
  - Added `--normalize_llm_hidden` with checkpoint-local `hidden_state_stats.npz`.
  - Added robust import fallback so direct script execution does not fail on missing `PYTHONPATH`.
- Updated `offline_rl/analyze_value_learning.py`:
  - Added same import fallback for robustness.

### Completed Diagnostics
- `6421542` (`value_diag_online_real`) COMPLETED
- `6421543` (`value_diag_online_zero`) COMPLETED
- Online JAX checkpoint (`online-jax-128env-skip100000000_20260221_153626.msgpack`) results:
  - hidden=`real`: Pearson `-0.0082`, EV `-0.0132`, MSE `8.1951`, MAE `2.1980`
  - hidden=`zero`: Pearson `-0.0398`, EV `-0.0050`, MSE `7.6557`, MAE `2.0844`

- `6421545` (`probe_realhidden_norm24_rl`) COMPLETED
  - Output: `logs/value_pairs_realhidden_norm24_rl.json`
  - Probe config: 24 base states, real LLM hidden vectors (normalized per checkpoint stats).
  - Mean sign-correct fraction across pair types:
    - `offline_pair_20260221_191229_real`: `0.715`
    - `offline_pair_20260221_191229_zero`: `0.750`

### Running Follow-ups (submitted, in progress)
- Checkpoint-curve diagnostics on exact-RTG runs:
  - `6421623` (`valcurve_real_hreal`)
  - `6421624` (`valcurve_zero_hreal`)
  - `6421625` (`valcurve_real_hzero`)
  - `6421626` (`valcurve_zero_hzero`)
- Exact-RTG real-hidden pair probe:
  - `6421692` (`probe_exactrtg_realhidden`)

---

## 2026-02-22: Clean Value Diagnostics Rerun + Interpretation

### Correctness Fixes Applied Before Rerun
- `offline_rl/analyze_value_learning.py` fixes:
  - Fixed dataset indexing bug: `files` and `counts` are now aligned by tracking `valid_files` only.
  - Added `--fail_on_any_error` so runs fail loudly if any checkpoint eval is `missing`/`error`.
  - Kept robust import fallback for direct-script/module invocation.
- Reason:
  - Initial curve jobs `6421623-6421626` finished but all checkpoint entries were `status=error` (`ModuleNotFoundError`), so they were treated as invalid/superseded.

### Exact-RTG Real-Hidden Pair Probe (Clean)
- `6421692` completed.
- Output: `logs/value_pairs_exactrtg_realhidden_6421692.json`
- Mean sign-correct over controlled pair types:
  - `new_exact_real`: `0.6875`
  - `new_exact_zero`: `0.7361`
- Interpretation:
  - On these counterfactual state-pairs, the zero-hidden-trained checkpoint remains slightly more monotonic than the real-hidden-trained checkpoint.

### Clean Curve Jobs (strict rerun)
- Jobs:
  - `6421726` `valcurve2_real_hreal`
  - `6421729` `valcurve2_real_hzero`
  - `6421727` `valcurve2_zero_hreal`
  - `6421728` `valcurve2_zero_hzero`
- Outputs:
  - `logs/value_curve2_real_hreal_6421726.json`
  - `logs/value_curve2_real_hzero_6421729.json`
  - `logs/value_curve2_zero_hreal_6421727.json`
  - `logs/value_curve2_zero_hzero_6421728.json`

#### Learning Trend (`new_exact_real`, eval hidden=`real`)
- 25k -> final:
  - Pearson: `0.3014 -> 0.3301`
  - EV: `0.0580 -> 0.0755`
  - Top-minus-bottom RTG: `2.1940 -> 2.4310`
  - MAE: `1.4973 -> 1.4846`
- Interpretation:
  - Value ranking quality improves steadily when evaluated under matching hidden-input conditions.

#### Hidden Dependence (`new_exact_real`, eval hidden=`zero`)
- Final metrics collapse relative to hidden=`real`:
  - Pearson `0.0795`, EV `0.0029`, top-minus-bottom `0.5390`.
- Interpretation:
  - The real-hidden-trained value head is strongly dependent on hidden input; ablating hidden degrades ranking sharply.

#### `new_exact_zero` behavior
- Eval hidden=`real` final:
  - Pearson `0.2929`, EV `0.0371`, top-minus-bottom `2.0644`.
- Eval hidden=`zero` final:
  - Pearson `0.1749`, EV `-0.0293`, top-minus-bottom `1.2707`.
- Interpretation:
  - This model is less robust overall as a calibrated value regressor; hidden-mode mismatch effects are non-trivial.

### Cross-Model Comparison (same exact-RTG sample)
- Jobs:
  - `6421742` (`valuecmp_hreal`)
  - `6421741` (`valuecmp_hzero`)
- Outputs:
  - `logs/value_compare_hreal_6421742.json`
  - `logs/value_compare_hzero_6421741.json`

#### hidden=`real`
- `old_offline_real`: Pearson `0.3316`, EV `0.1096`, MSE `6.5051`, bias `-1.7194`
- `old_offline_zero`: Pearson `0.3143`, EV `0.0988`, MSE `6.6229`, bias `-1.7411`
- `new_exact_real`: Pearson `0.3301`, EV `0.0755`, MSE `4.3714`, bias `-0.8288`
- `new_exact_zero`: Pearson `0.2929`, EV `0.0371`, MSE `4.5337`, bias `-0.8344`
- `online_jax_hidden`: Pearson `-0.0206`, EV `-0.0157`, MSE `9.6472`, bias `-2.3663`

#### hidden=`zero`
- `old_offline_real`: Pearson `0.2129`, EV `0.0160`, MSE `6.9781`
- `old_offline_zero`: Pearson `0.2245`, EV `0.0234`, MSE `5.3455`
- `new_exact_real`: Pearson `0.0795`, EV `0.0029`, MSE `5.6568`
- `new_exact_zero`: Pearson `0.1749`, EV `-0.0293`, MSE `4.5857`
- `online_jax_hidden`: Pearson `-0.0610`, EV `-0.0065`, MSE `9.0930`

### Main Takeaways
- Offline value heads are clearly better than the online JAX value head on this offline exact-RTG dataset slice.
- `new_exact_real` improves ordering under matching hidden input and has much lower MSE/bias than old offline checkpoints.
- Old checkpoints still show slightly higher EV under hidden=`real`, but with substantially worse calibration/bias.
- Hidden-input conditioning materially changes conclusions; hidden-mode at evaluation must match intended deployment assumptions.

### Operational Notes
- In this cluster, `SLURM_JOB_ID`/`SLURM_JOBID` was empty inside some `--wrap` commands; output JSONs were written with trailing `_`. Those were copied to job-id-stamped filenames above.
- One quick readability check job (`6421738`) failed due shell heredoc quoting in `sbatch --wrap`; this was a command-quoting issue, not a model/data run failure.

---

## TODO: Experiment Roadmap

### Priority 1: Offline RL AWR Training
- [ ] Generate fresh labelled dataset (PPO data → LLM hidden state annotation)
- [ ] Train AWR augmented model
- [ ] Train AWR baseline (no augmentation) for comparison

### Priority 2: Value Function Probing
Evaluate value function on curated observation states to understand what the LLM augmentation contributes:

- [ ] **Interesting game states**: arrow flying in, low on sleep, iron nearby, ladder nearby
- [ ] **Systematic prompt ablations**:
  - Arrow one block away vs far away vs no arrow
  - Pass different status values to LLM while keeping symbolic obs constant
  - Pass garbage/zero embeddings vs real embeddings
  - Pass embeddings from garbage prompts
- [ ] **LLM attention tests**:
  - If no enemy in frame but prompt says enemy → does value drop?
  - Pass nonsense text to the online-trained model
- [ ] **Compare augmented vs unaugmented value function** on same states
- [ ] Evaluate on frames from offline dataset AND novel frames

### Priority 2.5: Evaluation Infrastructure
- [ ] Create a flexible eval script + sbatch that:
  - Evaluates a policy with configurable LLM prompt (easy to swap prompts)
  - Tests whether the policy actually relies on the LLM
  - Supports running with modified/garbage prompts to probe LLM dependence
- [ ] Review existing eval scripts (`eval_awr.py`, `eval_awr_vlm_augmented.py`) and consolidate
- [ ] Consider: imagination/LLM augmentation should help with OOD states and exploration

### Priority 3: Scaling
- [ ] Try Qwen3-30B-A3B (MoE, ~3B active params, potentially better reasoning)

## 2026-02-23: Short Online/Offline Ablation Plan (128 envs)

### Goal
- Validate that augmented policies are not worse than unaugmented in short training windows.
- Specifically test whether a gated hidden-state fusion can default to near-unaugmented behavior early and then learn to use hidden states only if helpful.

### Modifications Under Test
- Added gated hidden fusion (`fusion_mode=gated_proj`) with learnable scalar gate initialized near zero (`hidden_gate_init_logit=-6.0`).
- Added `NO_LLM=1` support to `scripts/sbatch/run_online_rl_hidden_jax.sbatch` so unaugmented baseline can run in the same training stack without starting vLLM.

### Experiment Matrix (short runs)
- Online baseline: `NO_LLM=1`, 128 envs, short timesteps.
- Online augmented: `fusion_mode=gated_proj`, 128 envs, short timesteps, `skip_n=25`.
- Offline paired AWR: identical settings except hidden mode (`real` vs `zero`) under `fusion_mode=gated_proj`.

### Hypotheses
- If hidden states are unhelpful early, gated fusion should behave close to baseline (small gate, no major regression).
- If hidden states help, gate should increase and `real` should outperform `zero` in short training.
- If both fail to match baseline, issue is likely optimization/data pipeline mismatch rather than hidden-state quality alone.

### Submitted Runs (short ablations)
- Offline AWR (debug, 6h):
  - `6424082` (`awr_real_d`) hidden=`real`, `fusion_mode=gated_proj`, `hidden_gate_init_logit=-6.0`, `TOTAL_STEPS=200000`, `NUM_ENVS=128`, `MAX_FILES=256`.
  - `6424109` (`awr_zero_d`) hidden=`zero`, same settings as above (pending).
- Online JAX (debug, 6h):
  - `6424110` (`orlhj_base_d`) `NO_LLM=1`, `envs=128`, `timesteps=1_000_000`, checkpoint every `250k`.
  - `6424111` (`orlhj_gated_d`) `NO_LLM=0`, `skip_n=25`, `fusion_mode=gated_proj`, same training budget as baseline.

### Early Status
- `6424082` is running and loading dataset with automatic file limiting:
  - Estimated 84.62 GiB for 256 files; auto-limited to 60 files / 491,520 samples under 20.0 GiB budget.
  - No runtime errors so far; WandB run initialized (`kq5gkllk`).
- Other short jobs are currently queued by scheduler priority.

### Live Observations (early)
- `awr_real_d` (`6424082`) and `awr_zero_d` (`6424109`) are both running stably (no crashes).
- Early optimization trend:
  - `real` run latest observed: `Step 31000/200000`, `expl_var~0.909`, `gate~0.0309`.
  - `zero` run latest observed: `Step 89000/200000`, `expl_var~0.929`, `gate~0.0012`.
- Interpretation (tentative):
  - Gate remains very small for `zero` hidden mode (model keeps hidden path effectively suppressed).
  - Gate grows for `real` hidden mode, indicating model is learning to use hidden-state pathway when signal exists.

### Online Short-Run Scheduling
- Submitted GPU online jobs (`6424110`, `6424111`) are queued with reason `QOSMaxGRESPerUser` due per-user GPU quota.
- To keep progress moving, launched CPU no-LLM online baseline:
  - `6424124` (`orlhj_base_cpu`), `envs=128`, `timesteps=200000`, WandB run `2moa8czt`.
  - Startup and JAX CPU fallback confirmed; currently in long initial compile/update phase.

### Follow-up Results (same day)
- `6424109` (`awr_zero_d`) completed successfully in `00:13:14`.
  - Final checkpoint: `/data/group_data/rl/geney/checkpoints/awr_llm_augmented_live/shortgatedd_20260222_200219_zero/awr_llm_final.pth`
  - Gate remained near-zero throughout (`~0.0012-0.0013`), consistent with hidden-path suppression.
- `6424082` (`awr_real_d`) is still running and currently at `Step 122000/200000`.
  - Checkpoints present at `50k` and `100k`.
  - Gate has increased materially (`~0.29` at 122k), indicating learned hidden-state usage.

### Online Short Runs
- `6424124` (`orlhj_base_cpu`) completed (`200k` steps, no-LLM baseline).
  - Throughput was low on CPU (`~348 SPS`), final logged return around `3.5`.
- `6424110` (`orlhj_base_d`) completed (`1M` steps, no-LLM on GPU).
  - Intermediate checkpoints saved at `~254k`, `~508k`, `~754k` and final at `~999k`.
  - Return trend: `2.8 -> 5.1 -> 4.9` (updates 10 -> 90 -> final).
- `6424111` (`orlhj_gated_d`) started after quota freed and is currently in vLLM model-load phase.

### Analysis Fix + New Diagnostics
- Fixed checkpoint-loading incompatibility in:
  - `offline_rl/analyze_value_learning.py`
  - `offline_rl/analyze_value_pairs.py`
- Change: analyzer now auto-detects fusion mode (`gated_proj` vs `concat_raw`) from state_dict keys before model instantiation.
- New outputs:
  - `logs/value_short_ablation_hiddenreal_v2.json`
  - `logs/value_short_ablation_hiddenzero_v2.json`
- Key metric snapshot (`Pearson / EV / MAE`):
  - `real ckpt 50k` eval hidden=`real`: `0.954 / 0.911 / 0.182`
  - `real ckpt 100k` eval hidden=`real`: `0.973 / 0.946 / 0.142`
  - `real ckpt 100k` eval hidden=`zero`: `0.903 / 0.815 / 0.270`
  - `zero ckpt 150k` eval hidden=`real`: `0.972 / 0.945 / 0.142`
  - `zero ckpt 150k` eval hidden=`zero`: `0.972 / 0.945 / 0.142`
- Interpretation:
  - Real-hidden model quality improves strongly with training under matching hidden input.
  - Real-hidden model degrades under hidden-zero ablation, consistent with genuine hidden dependence.
  - Zero-hidden model is effectively hidden-invariant (nearly identical metrics under hidden real/zero eval).

## 2026-02-23: Symbolic Policy Eval Suite (W&B project `craftax_symbolic_evals`)

### What was added
- New evaluator script: `scripts/eval_symbolic_policy_suite.py`
  - Evaluates symbolic policies with `128` episodes each.
  - Logs return/length/achievement stats + one rollout video to W&B.
  - Logs a model/training summary block per run.
  - Supports selecting policy subset via `--policies`.
- New launcher: `scripts/sbatch/run_symbolic_policy_evals.sbatch`
  - Handles W&B logging, batch defaults, and optional vLLM startup.
  - Supports `START_VLLM=0` for non-LLM policies (used for PPO-only rerun).

### Robustness/cluster fixes applied during eval setup
- `scripts/start_vllm_hidden.sh`:
  - Stage vLLM config to `/scratch`.
  - Move temp/cache/pycache/HF/vLLM caches to `/scratch`.
  - Set stable served model name (`--served-model-name ./configs/vllm_hidden_qwen4b`) so extractor model lookup still works after staging.
  - Lower default max model len to `4096` for faster startup.
- `scripts/sbatch/run_symbolic_policy_evals.sbatch`:
  - Added `/scratch` diagnostics and optional vLLM startup gate.

### Final eval runs (completed)
- `skip25_real_hidden` run id `fg7zbpat`
  - Mean return `3.8813`, return std `1.1722`
  - Mean achievements `4.7813`
  - 128 episodes, env steps `374`, runtime `150.36s`
  - Logged with real hidden-state extraction (`llm_calls=15`)
- `skip100m_baseline` run id `g19ip9eq`
  - Mean return `1.3891`, return std `1.2321`
  - Mean achievements `2.2891`
  - 128 episodes, env steps `376`, runtime `117.55s`
  - Zero-hidden eval mode (`llm_calls=0`)
- `ppo_symbolic` run id `mmucw5ls`
  - Mean return `18.6312`, return std `2.7327`
  - Mean achievements `19.1563`
  - 128 episodes, env steps `936`, runtime `97.93s`

### Artifacts
- Combined summary JSON:
  - `/home/geney/Craftax_Baselines/logs/policy_eval_craftax_symbolic_evals_combined_20260223.json`
- PPO-only evaluator JSON:
  - `/home/geney/Craftax_Baselines/logs/policy_eval_craftax_symbolic_evals_6425376.json`

### Key interpretation
- The augmented online policies are underperforming *during training itself* (not only at eval):
  - Skip25 training final return (logged): `~4.08`
  - Skip100M training final return (logged): `~3.06`
  - PPO reference training final return (logged): `~19.67`
- Therefore the primary issue is a training-stack mismatch / optimization problem relative to PPO baseline, not just hidden-state availability at evaluation.

## 2026-02-24: Labelling Tail-End Reliability + Offline Chain Hardening

### Root cause observed
- Final labelling stage repeatedly re-spawned single workers for one missing file.
- Primary hard failure:
  - `OSError: [Errno 122] Disk quota exceeded`
  - Triggered by `labelling/llm_worker.py` creating file logs under:
    - `/data/group_data/rl/geney/craftax_llm_job_logs/...`

### Fixes applied
- `labelling/llm_worker.py`
  - Logging is now quota-safe:
    - always logs to stdout,
    - tries file logging at `LOGS_DIR`,
    - falls back to local scratch (`WORKER_LOCAL_DIR/worker_logs`) if shared path fails,
    - continues without file log if both fail.
  - Default `LOGS_DIR` now points to local worker storage:
    - `LOGS_DIR=${WORKER_LOCAL_DIR}/worker_logs` (unless overridden).
- `labelling/run_finish_remaining_once.sbatch`
  - Added multi-round completion logic:
    - rebuilds missing set each round,
    - exits non-zero if final missing count > 0.
  - Added worker script override:
    - `WORKER_SCRIPT=...` (used to run finishing workers on `rl` partition).
  - Added rescue controls to avoid queue storms:
    - `MAX_RESCUES_PER_ROUND`
    - `RESCUE_COOLDOWN_SEC`
  - Default `LOGS_DIR` changed to scratch:
    - `/scratch/$USER/craftax_llm_job_logs`
- `labelling/makeworkers_llm_rl.sbatch`
  - Added RL-partition worker launcher for non-preempt tail completion.
  - Uses `--partition=rl` and `--gres=gpu:1`.

### Operational result
- Tail-end worker now runs stably on `rl` and processes the remaining file without log-quota crashes.
- Dependency chain is guarded and sequential:
  1. `llm_finish_once`
  2. `run_recompute_rtg_exact.sbatch`
  3. `run_offline_awr_llm_augmented.sbatch` x3 (`concat_raw`, `gated_proj`, `residual_gated`)

### Active chain (at handoff)
- Finish job: `6436636`
- RTG job (after finish): `6436638`
- AWR jobs (after RTG): `6436639`, `6436640`, `6436641`

### Quick runbook
- Tail-only finish on RL:
  - `WORKER_SCRIPT=/home/geney/Craftax_Baselines/labelling/makeworkers_llm_rl.sbatch NUM_WORKERS=1 sbatch labelling/run_finish_remaining_once.sbatch`
- Monitor:
  - `tail -f /home/geney/Craftax_Baselines/logs/label_finish_once_<jobid>.out`
  - `squeue -u geney`
- Validate completion:
  - compare counts of `trajectories_batch_*.npz` in source vs labelled results.
