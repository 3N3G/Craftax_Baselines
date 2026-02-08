#!/usr/bin/env python3
"""
Online RL Smoke Test with SGLang

Runs a minimal online RL loop with LLM policy to validate the setup and
estimate training throughput.

Usage:
    # Start SGLang server first:
    python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Thinking-2507 \
        --port 30000 --enable-radix-caching --dtype float16
    
    # Run smoke test:
    python online_rl_smoke_test.py --steps 100 --envs 4
"""

import argparse
import os
import sys
import time

# Prevent JAX from hogging GPU memory (SGLang needs it)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_text

from sglang_policy import SGLangPolicy


def run_smoke_test(
    num_envs: int = 4,
    num_steps: int = 100,
    server_url: str = "http://localhost:30000",
    verbose: bool = True,
):
    """Run a minimal online RL loop with LLM policy."""
    
    print("=" * 60)
    print("Online RL Smoke Test with SGLang")
    print("=" * 60)
    
    # Initialize policy
    policy = SGLangPolicy(server_url=server_url)
    
    if not policy.check_health():
        print(f"\n❌ ERROR: SGLang server not healthy at {server_url}")
        print("\nStart the server with:")
        print("  python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Thinking-2507 \\")
        print("      --port 30000 --enable-radix-caching --dtype float16")
        return None
    
    print(f"✓ SGLang server healthy at {server_url}")
    
    # Initialize environment
    print(f"\nInitializing {num_envs} Craftax environments...")
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    
    # Vectorized reset
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, num_envs)
    
    # Reset all environments
    obs_list = []
    state_list = []
    for i, r in enumerate(rngs):
        obs, state = env.reset(r, env_params)
        obs_list.append(obs)
        state_list.append(state)
    
    print(f"✓ Environments initialized")
    
    # Run loop
    print(f"\nRunning {num_steps} steps with batch size {num_envs}...")
    print("-" * 60)
    
    total_reward = 0.0
    episode_rewards = []
    current_episode_rewards = [0.0] * num_envs
    step_times = []
    
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        
        # Render observations to text
        text_obs = []
        for state in state_list:
            text = render_craftax_text(state)
            # Simple filtering (like llm_play_harnessed.py)
            lines = text.split("\n")
            filtered = []
            for line in lines:
                if any(skip in line.lower() for skip in ["grass", "sand", "gravel"]):
                    continue
                filtered.append(line)
            text_obs.append("\n".join(filtered[:50]))  # Truncate for speed
        
        # Get actions from LLM
        actions = policy.get_actions_batch(text_obs)
        
        # Step environments
        new_obs_list = []
        new_state_list = []
        step_rewards = []
        
        for i, (state, action) in enumerate(zip(state_list, actions)):
            rng, step_rng = jax.random.split(rngs[i])
            rngs = rngs.at[i].set(rng)
            
            obs, new_state, reward, done, info = env.step(
                step_rng, state, int(action), env_params
            )
            
            step_rewards.append(float(reward))
            current_episode_rewards[i] += float(reward)
            
            if done:
                episode_rewards.append(current_episode_rewards[i])
                current_episode_rewards[i] = 0.0
                # Auto-reset handled by env
            
            new_obs_list.append(obs)
            new_state_list.append(new_state)
        
        state_list = new_state_list
        total_reward += sum(step_rewards)
        
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        
        if verbose and (step + 1) % 10 == 0:
            avg_step_time = np.mean(step_times[-10:])
            samples_per_sec = num_envs / avg_step_time
            print(f"  Step {step+1:4d}/{num_steps} | "
                  f"Reward: {sum(step_rewards):.1f} | "
                  f"Eps completed: {len(episode_rewards)} | "
                  f"Step time: {avg_step_time:.2f}s | "
                  f"Samples/s: {samples_per_sec:.2f}")
    
    elapsed = time.perf_counter() - start_time
    
    # Summary
    print("-" * 60)
    print("\n📊 RESULTS")
    print("=" * 60)
    
    metrics = policy.get_metrics()
    total_samples = num_envs * num_steps
    
    results = {
        "total_steps": num_steps,
        "num_envs": num_envs,
        "total_samples": total_samples,
        "elapsed_time_s": elapsed,
        "samples_per_sec": total_samples / elapsed,
        "steps_per_sec": num_steps / elapsed,
        "avg_step_time_s": np.mean(step_times),
        "total_reward": total_reward,
        "episodes_completed": len(episode_rewards),
        "avg_episode_reward": np.mean(episode_rewards) if episode_rewards else 0,
        "policy_failure_rate": metrics["failure_rate"],
    }
    
    print(f"  Total samples:     {results['total_samples']}")
    print(f"  Elapsed time:      {results['elapsed_time_s']:.1f}s")
    print(f"  Samples/sec:       {results['samples_per_sec']:.2f}")
    print(f"  Episodes completed: {results['episodes_completed']}")
    print(f"  Avg episode reward: {results['avg_episode_reward']:.2f}")
    print(f"  Policy failures:   {metrics['failed_requests']}/{metrics['total_requests']} ({metrics['failure_rate']*100:.1f}%)")
    
    # Training time estimates
    print("\n⏱️  TRAINING TIME ESTIMATES")
    print("-" * 60)
    sps = results['samples_per_sec']
    
    for target in [1e6, 10e6, 50e6, 100e6, 1e9]:
        time_hours = target / sps / 3600
        time_days = time_hours / 24
        status = "✅" if time_days <= 3.5 else "⚠️" if time_days <= 7 else "❌"
        print(f"  {target/1e6:6.0f}M steps: {time_hours:8.1f} hours ({time_days:5.1f} days) {status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Online RL smoke test with SGLang")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel envs")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--server-url", type=str, default="http://localhost:30000")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    results = run_smoke_test(
        num_envs=args.envs,
        num_steps=args.steps,
        server_url=args.server_url,
        verbose=not args.quiet,
    )
    
    if results:
        print("\n✅ Smoke test completed successfully!")
        print(f"\nTo run with more parallelism:")
        print(f"  python online_rl_smoke_test.py --envs 64 --steps 100")
    else:
        print("\n❌ Smoke test failed. Check server status.")
        sys.exit(1)


if __name__ == "__main__":
    main()
