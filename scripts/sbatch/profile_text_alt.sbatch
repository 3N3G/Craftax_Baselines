#!/bin/bash
#SBATCH --job-name=profile_text_alt
#SBATCH --partition=rl
#SBATCH --gres=gpu:RTX_PRO_6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/profile_text_alt_%j.out
#SBATCH --error=logs/profile_text_alt_%j.err

set -e
source ~/.bashrc

cd ~/Craftax_Baselines
mkdir -p logs

echo "========================================"
echo "Profile text path alternatives"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --list-gpus | head -1)"
echo "========================================"

conda activate /data/user_data/geney/.conda/envs/craftax

python - <<'PY'
import re
import time
import numpy as np
import jax

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_text
from utils.wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper
from utils.llm_prompts import filter_text_obs
from labelling.obs_to_text import obs_to_text

NUM_ENVS = 8
ITERS = 20
coord_pattern = re.compile(r"^(-?\d+),\s*(-?\d+):")


def render_craftax_text_swapped(state):
    st = render_craftax_text(state)
    lines = st.split("\n")
    out = []
    for line in lines:
        m = coord_pattern.match(line)
        if m:
            col, row = m.groups()
            out.append(line.replace(f"{col}, {row}:", f"{row}, {col}:", 1))
        else:
            out.append(line)
    return "\n".join(out)


env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params
env = LogWrapper(env)
env = AutoResetEnvWrapper(env)
env = BatchEnvWrapper(env, num_envs=NUM_ENVS)

rng = jax.random.PRNGKey(0)
rng, rr = jax.random.split(rng)
obs, env_state = env.reset(rr, env_params)

rng, ar = jax.random.split(rng)
actions = jax.random.randint(ar, (NUM_ENVS,), 0, env.action_space(env_params).n)
rng, sr = jax.random.split(rng)
obs, env_state, rew, done, info = env.step(sr, env_state, actions, env_params)

# Keep host copy for fair text-path timing
obs_host = np.asarray(jax.device_get(obs))

legacy_ms = []
obs_to_text_ms = []

for _ in range(ITERS):
    # Legacy path: render from env_state
    t0 = time.perf_counter()
    for i in range(NUM_ENVS):
        s = jax.tree.map(lambda x: x[i], env_state.env_state)
        raw = render_craftax_text_swapped(s)
        _ = filter_text_obs(raw)
    legacy_ms.append((time.perf_counter() - t0) * 1000.0)

    # Alternative path: decode from symbolic obs vector
    t1 = time.perf_counter()
    for i in range(NUM_ENVS):
        raw2 = obs_to_text(obs_host[i])
        _ = filter_text_obs(raw2)
    obs_to_text_ms.append((time.perf_counter() - t1) * 1000.0)

print("backend:", jax.default_backend())
print("devices:", jax.devices())
print("legacy_ms_mean:", float(np.mean(legacy_ms)))
print("legacy_ms_p95:", float(np.percentile(legacy_ms, 95)))
print("obs_to_text_ms_mean:", float(np.mean(obs_to_text_ms)))
print("obs_to_text_ms_p95:", float(np.percentile(obs_to_text_ms, 95)))
print("speedup_legacy_over_obs_to_text:", float(np.mean(legacy_ms) / max(1e-9, np.mean(obs_to_text_ms))))
PY
