#!/usr/bin/env python3
"""
Microbenchmark for online RL hidden-state extraction.

This isolates the hot path used in online_rl_hidden_jax.py:
1) render/filter text for each env
2) request hidden states from vLLM
"""

import argparse
import os
import sys
import time
from typing import List

import jax
import numpy as np

# Make repo-local imports (`utils`, `online_rl_llm`, etc.) work when this script
# is run as `python scripts/diagnostics/bench_hidden_extractor.py`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from craftax.craftax_env import make_craftax_env_from_name
from utils.wrappers import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper
from online_rl_llm.online_rl_hidden_jax import Config, LLMHiddenStateManager


def _format_stats(values: List[float], label: str) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"{label}_mean={arr.mean():.2f} "
        f"{label}_p50={np.percentile(arr, 50):.2f} "
        f"{label}_p95={np.percentile(arr, 95):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark hidden-state extraction path")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--model", type=str, default=Config.MODEL_ID)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("Hidden Extractor Microbenchmark", flush=True)
    print("=" * 70, flush=True)
    print(f"envs={args.envs} iters={args.iters} warmup={args.warmup}", flush=True)
    print(f"tokens={args.tokens} layer={args.layer} model={args.model}", flush=True)

    env = make_craftax_env_from_name(Config.ENV_NAME, auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=args.envs)

    llm_manager = LLMHiddenStateManager(
        model_id=args.model, target_layer=args.layer, tokens_to_generate=args.tokens
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)
    action_dim = env.action_space(env_params).n

    total_ms, text_ms, llm_ms = [], [], []
    n_total = args.warmup + args.iters

    for i in range(n_total):
        t0 = time.perf_counter()
        _, metrics = llm_manager.extract(obs, args.envs)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        rng, action_rng = jax.random.split(rng)
        actions = jax.random.randint(action_rng, (args.envs,), 0, action_dim)
        rng, step_rng = jax.random.split(rng)
        obs, env_state, _, _, _ = env.step(step_rng, env_state, actions, env_params)

        if i < args.warmup:
            continue

        text_i = float(metrics.get("timing/text_render_ms", 0.0))
        llm_i = float(metrics.get("timing/llm_inference_ms", 0.0))
        total_ms.append(dt_ms)
        text_ms.append(text_i)
        llm_ms.append(llm_i)

        approx_sps = args.envs / max(1e-9, dt_ms / 1000.0)
        print(
            f"ITER {i - args.warmup + 1:02d} "
            f"total_ms={dt_ms:.2f} text_ms={text_i:.2f} llm_ms={llm_i:.2f} "
            f"approx_sps={approx_sps:.2f}",
            flush=True,
        )

    mean_total_s = np.mean(total_ms) / 1000.0
    approx_sps_mean = args.envs / max(1e-9, mean_total_s)
    print(
        "SUMMARY "
        + _format_stats(total_ms, "extract_ms")
        + " "
        + _format_stats(text_ms, "text_ms")
        + " "
        + _format_stats(llm_ms, "llm_ms")
        + f" approx_sps={approx_sps_mean:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
