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
