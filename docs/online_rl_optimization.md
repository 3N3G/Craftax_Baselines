# Online RL Optimization: Architecture Analysis & Instructions

## Goal

Optimize `online_rl_llm/online_rl_hidden_jax.py` to match the baseline PPO performance of **~18,500 SPS** (samples per second). Currently achieves **~750 SPS** on CPU.

---

## Background

### What This Agent Does

The online RL agent plays the Craftax game while using an LLM (Qwen3-4B via vLLM) to extract hidden state representations from text observations. The policy network receives both the symbolic observation AND the LLM's hidden state as input.

### Current Architecture

```
For each step:
  1. Python loop: render text observations for all 128 envs
  2. HTTP call to vLLM server: extract hidden states (~100ms)
  3. JAX: policy forward pass → actions
  4. JAX: env.step → new states
  5. PPO update every N steps
```

**Why it's slow (750 SPS):**
- Runs on **CPU** (`JAX_PLATFORM_NAME=cpu`) to avoid GPU conflicts with vLLM
- Uses a **Python for-loop** for the rollout (no JIT compilation)
- Every step calls back to Python for text rendering, numpy conversions, etc.

### Baseline PPO Architecture (18,500 SPS)

See [online_rl/ppo.py](file:///Users/gene/Documents/Craftax_Baselines/online_rl/ppo.py).

**Why it's fast:**
1. **`BatchEnvWrapper`** auto-vectorizes `env.reset`/`env.step` across all envs
2. **`jax.lax.scan`** compiles the entire rollout + update into a single GPU kernel — zero Python overhead
3. **GPU execution** — everything runs on GPU

Key code pattern (ppo.py lines 242-344):
```python
def _update_step(runner_state, unused):
    def _env_step(runner_state, unused):
        # SELECT ACTION (JIT-compiled)
        pi, value = network.apply(train_state.params, last_obs)
        action = pi.sample(seed=_rng)
        
        # STEP ENV (JIT-compiled)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        
        return runner_state, transition
    
    # Entire rollout is JIT-compiled via scan
    runner_state, traj_batch = jax.lax.scan(
        _env_step, runner_state, None, config["NUM_STEPS"]
    )
    # ... GAE + PPO update also JIT-compiled ...

# Outer loop also JIT-compiled
runner_state, metric = jax.lax.scan(
    _update_step, runner_state, None, config["NUM_UPDATES"]
)
```

---

## Optimization Plan

### Core Idea: Two-Phase Loop

Split the training loop into **two phases**:

- **Phase A (Python, slow, ~100ms):** Text rendering + vLLM hidden state extraction. Runs every `skip_n` steps.
- **Phase B (JIT-compiled, fast, ~microseconds/step):** Run `skip_n` steps of env+policy via `jax.lax.scan`. Zero Python overhead.

```python
# Pseudocode:
for outer_step in range(total_steps // skip_n):
    # Phase A: LLM inference (Python, ~100ms)
    text_obs = render_text_for_all_envs(states)  # Python loop
    hidden = extract_from_vllm(text_obs)          # HTTP call
    
    # Phase B: Run skip_n steps fully JIT-compiled
    (states, obs, metrics), traj = jax.lax.scan(
        jit_env_step, (states, obs, hidden, params), None, skip_n
    )
```

With `skip_n=1`, every step needs LLM → no speedup from scan.
With `skip_n=4+`, inner steps run at near-18,500 SPS between LLM calls.
With `skip_n=∞` (no LLM), should match baseline PPO exactly.

### Required Changes

#### 1. Remove CPU-only mode

Currently line 22:
```python
os.environ["JAX_PLATFORM_NAME"] = "cpu"
```

**Change to:**
```python
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Don't grab all GPU memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"   # Use only 30% of GPU for JAX
# Let vLLM manage its own 95% via --gpu-memory-utilization 0.95
```

JAX and vLLM can coexist on the same GPU — they use separate memory allocators (XLA vs PyTorch/CUDA).

#### 2. Use BatchEnvWrapper

Currently the JAX version manually manages env state. Use the same wrappers as PPO:

```python
from utils.wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper

env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env = LogWrapper(env)
env = AutoResetEnvWrapper(env)
env = BatchEnvWrapper(env, num_envs=128)
```

#### 3. JIT-compile the inner rollout via `jax.lax.scan`

Create a pure function for the env+policy step (no Python IO):

```python
@jax.jit
def rollout_n_steps(carry, unused):
    """Single env+policy step — fully JIT-compiled."""
    train_state, env_state, last_obs, hidden_states, rng = carry
    
    # Policy forward: obs + hidden → action
    rng, _rng = jax.random.split(rng)
    pi, value = network.apply(train_state.params, last_obs, hidden_states)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)
    
    # Env step
    rng, _rng = jax.random.split(rng)
    next_obs, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
    
    transition = Transition(done, action, value, reward, log_prob, last_obs, next_obs, info)
    carry = (train_state, env_state, next_obs, hidden_states, rng)
    return carry, transition
```

Then in the main loop:
```python
# Run skip_n steps with fixed hidden states
carry, traj_batch = jax.lax.scan(rollout_n_steps, carry, None, skip_n)

# PPO update (also JIT-compiled, copy from ppo.py)
# ... calculate GAE, update network ...
```

#### 4. Text rendering only when needed

Text rendering (128 calls to `render_craftax_text`) happens in Python and takes ~50ms. Only do this in Phase A:

```python
# Phase A: every skip_n steps
if step % skip_n == 0:
    # This is the ONLY Python code in the entire loop
    text_obs = [render_craftax_text(state) for state in env_states]
    filtered = [filter_text_obs(t) for t in text_obs]
    hidden, metrics = extractor.extract_hidden_states_no_cot(filtered)
    hidden_jax = jnp.array(hidden)  # Convert to JAX array for Phase B
```

### Key Files to Study

| File | What to learn |
|------|--------------|
| [online_rl/ppo.py](file:///Users/gene/Documents/Craftax_Baselines/online_rl/ppo.py) | Reference JIT-compiled rollout architecture (lines 90-780) |
| [online_rl_llm/online_rl_hidden_jax.py](file:///Users/gene/Documents/Craftax_Baselines/online_rl_llm/online_rl_hidden_jax.py) | Current implementation to optimize (647 lines) |
| [online_rl_llm/online_rl_hidden.py](file:///Users/gene/Documents/Craftax_Baselines/online_rl_llm/online_rl_hidden.py) | PyTorch version for reference (has correct training logic) |
| [models/actor_critic.py](file:///Users/gene/Documents/Craftax_Baselines/models/actor_critic.py) | `ActorCriticAug` model (takes obs + hidden_state) |
| [utils/wrappers.py](file:///Users/gene/Documents/Craftax_Baselines/utils/wrappers.py) | `BatchEnvWrapper`, `AutoResetEnvWrapper`, `LogWrapper` |
| [utils/llm_extractor.py](file:///Users/gene/Documents/Craftax_Baselines/utils/llm_extractor.py) | `VLLMHiddenStateExtractor` — HTTP client to vLLM server |
| [scripts/start_vllm_hidden.sh](file:///Users/gene/Documents/Craftax_Baselines/scripts/start_vllm_hidden.sh) | How to start the vLLM server |
| [scripts/sbatch/run_online_rl_hidden_jax.sbatch](file:///Users/gene/Documents/Craftax_Baselines/scripts/sbatch/run_online_rl_hidden_jax.sbatch) | SLURM submission script (starts vLLM + runs agent) |

### Key Constraints

1. **vLLM server must be running** — it serves hidden states via HTTP on `localhost:8000`
2. **`render_craftax_text`** requires the raw JAX env state — can't be JIT-compiled (Python string manipulation)
3. **Hidden states are fixed between LLM calls** — during the `skip_n` inner steps, the policy uses the same hidden state from the last LLM call
4. **The `ActorCriticAug` model** in `models/actor_critic.py` takes `(obs, hidden_state)` as input — use `ActorCritic` for plain obs-only

### Performance Expectations

| Configuration | Expected SPS | Notes |
|--------------|-------------|-------|
| Current (CPU, Python loop) | ~750 | Baseline measurement |
| GPU + `jax.lax.scan`, no LLM | ~18,500 | Should match ppo.py |
| GPU + `jax.lax.scan`, skip_n=100 | ~15,000-17,000 | LLM call ~100ms every 100 steps |
| GPU + `jax.lax.scan`, skip_n=25 | ~10,000-14,000 | LLM call ~100ms every 25 steps |
| GPU + `jax.lax.scan`, skip_n=1 | ~1,000-2,000 | LLM every step, scan overhead minimal |

### Verification

1. **Correct behavior:** Run with `--steps 100 --skip-n 1000000 --no-wandb` and compare SPS with PPO baseline
2. **LLM integration:** Run with `--steps 100 --skip-n 25` and verify hidden states are extracted correctly
3. **Training quality:** Compare learning curves on WandB between the optimized JAX version and the PyTorch version

---

## Related Documentation

- [CLAUDE.md](file:///Users/gene/Documents/Craftax_Baselines/CLAUDE.md) — Main project reference
- [docs/progress_journal.md](file:///Users/gene/Documents/Craftax_Baselines/docs/progress_journal.md) — Experiment log
- [docs/llm_labelling.md](file:///Users/gene/Documents/Craftax_Baselines/docs/llm_labelling.md) — Labelling pipeline docs
