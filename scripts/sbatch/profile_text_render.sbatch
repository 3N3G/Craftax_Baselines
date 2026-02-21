#!/bin/bash
#SBATCH --job-name=profile_text_render
#SBATCH --partition=rl
#SBATCH --gres=gpu:RTX_PRO_6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/profile_text_render_%j.out
#SBATCH --error=logs/profile_text_render_%j.err

set -e
source ~/.bashrc

cd ~/Craftax_Baselines
mkdir -p logs

echo "========================================"
echo "Profile text rendering path"
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

NUM_ENVS = 8
ITERS = 10

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


def old_path(env_states, num_envs):
    t_slice = t_render = t_filter = 0.0
    for i in range(num_envs):
        a = time.perf_counter()
        single_state = jax.tree.map(lambda x: x[i], env_states)
        t_slice += time.perf_counter() - a

        b = time.perf_counter()
        raw = render_craftax_text_swapped(single_state)
        t_render += time.perf_counter() - b

        c = time.perf_counter()
        _ = filter_text_obs(raw)
        t_filter += time.perf_counter() - c
    return t_slice * 1000, t_render * 1000, t_filter * 1000


def host_once_path(env_states, num_envs):
    t_get = t_slice = t_render = t_filter = 0.0

    a = time.perf_counter()
    host_states = jax.device_get(env_states)
    t_get += time.perf_counter() - a

    for i in range(num_envs):
        b = time.perf_counter()
        single_state = jax.tree.map(lambda x: x[i], host_states)
        t_slice += time.perf_counter() - b

        c = time.perf_counter()
        raw = render_craftax_text_swapped(single_state)
        t_render += time.perf_counter() - c

        d = time.perf_counter()
        _ = filter_text_obs(raw)
        t_filter += time.perf_counter() - d
    return t_get * 1000, t_slice * 1000, t_render * 1000, t_filter * 1000


env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params
env = LogWrapper(env)
env = AutoResetEnvWrapper(env)
env = BatchEnvWrapper(env, num_envs=NUM_ENVS)

rng = jax.random.PRNGKey(0)
rng, rr = jax.random.split(rng)
obs, env_state = env.reset(rr, env_params)

# step once to avoid first-call oddities
rng, ar = jax.random.split(rng)
actions = jax.random.randint(ar, (NUM_ENVS,), 0, env.action_space(env_params).n)
rng, sr = jax.random.split(rng)
obs, env_state, rew, done, info = env.step(sr, env_state, actions, env_params)

old_totals = []
new_totals = []
old_parts = []
new_parts = []

for _ in range(ITERS):
    s_ms, r_ms, f_ms = old_path(env_state.env_state, NUM_ENVS)
    old_parts.append((s_ms, r_ms, f_ms))
    old_totals.append(s_ms + r_ms + f_ms)

    g_ms, s2_ms, r2_ms, f2_ms = host_once_path(env_state.env_state, NUM_ENVS)
    new_parts.append((g_ms, s2_ms, r2_ms, f2_ms))
    new_totals.append(g_ms + s2_ms + r2_ms + f2_ms)

print("backend:", jax.default_backend())
print("devices:", jax.devices())
print("--- old path (per-env jax.tree.map on device arrays) ---")
print("total_ms_mean:", float(np.mean(old_totals)))
print("slice_ms_mean:", float(np.mean([x[0] for x in old_parts])))
print("render_ms_mean:", float(np.mean([x[1] for x in old_parts])))
print("filter_ms_mean:", float(np.mean([x[2] for x in old_parts])))

print("--- host-once path (jax.device_get once, then slice host arrays) ---")
print("total_ms_mean:", float(np.mean(new_totals)))
print("device_get_ms_mean:", float(np.mean([x[0] for x in new_parts])))
print("slice_ms_mean:", float(np.mean([x[1] for x in new_parts])))
print("render_ms_mean:", float(np.mean([x[2] for x in new_parts])))
print("filter_ms_mean:", float(np.mean([x[3] for x in new_parts])))

speedup = float(np.mean(old_totals) / max(1e-9, np.mean(new_totals)))
print("speedup_old_over_new:", speedup)
PY
