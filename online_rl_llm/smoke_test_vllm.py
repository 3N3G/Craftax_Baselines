#!/usr/bin/env python3
"""
Online RL Smoke Test with vLLM

Runs a minimal online RL loop with vLLM policy to validate setup and
estimate training throughput.

Uses identical observation filtering to llm_play_harnessed.py.

Usage:
    python online_rl_vllm_test.py --steps 50 --envs 8
"""

import argparse
import os
import re
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np


# =============================================================================
# Observation Processing (from llm_play_harnessed.py)
# =============================================================================

BACKGROUND_TILES = {
    "grass", "sand", "gravel", 
    "fire grass", "ice grass", "fire_grass", "ice_grass"
}


def filter_text_obs(text_obs: str) -> str:
    """
    Filter out background tiles from the text observation.
    Identical to llm_play_harnessed.py lines 101-172.
    """
    lines = text_obs.split('\n')
    filtered_lines = []
    in_map_section = False
    interesting_tiles = []
    
    for line in lines:
        stripped = line.strip()
        
        if stripped == 'Map:':
            in_map_section = True
            interesting_tiles = []
            continue
        
        if in_map_section:
            if ':' in stripped and ',' in stripped.split(':')[0]:
                parts = stripped.split(':', 1)
                if len(parts) == 2:
                    coord = parts[0].strip()
                    tile = parts[1].strip().lower()
                    
                    is_background = tile in BACKGROUND_TILES
                    has_entity = ' on ' in tile
                    
                    if not is_background or has_entity:
                        interesting_tiles.append(f"{coord}:{parts[1].strip()}")
                continue
            else:
                in_map_section = False
                if interesting_tiles:
                    filtered_lines.append(f"Map (interesting tiles only): {', '.join(interesting_tiles)}")
                else:
                    filtered_lines.append("Map: [No interesting tiles in view - all background]")
                if stripped:
                    filtered_lines.append(line)
                continue
        
        if stripped:
            filtered_lines.append(line)
    
    if in_map_section:
        if interesting_tiles:
            filtered_lines.append(f"Map (interesting tiles only): {', '.join(interesting_tiles)}")
        else:
            filtered_lines.append("Map: [No interesting tiles in view - all background]")
    
    return '\n'.join(filtered_lines)


def render_craftax_text_swapped(state):
    """
    Render text and swap Col,Row to Row,Col.
    From llm_play_harnessed.py lines 643-662.
    """
    from craftax.craftax.renderer import render_craftax_text
    
    st = render_craftax_text(state)
    lines = st.split('\n')
    new_lines = []
    
    coord_pattern = re.compile(r"^(-?\d+),\s*(-?\d+):")
    
    for line in lines:
        match = coord_pattern.match(line)
        if match:
            col, row = match.groups()
            new_line = line.replace(f"{col}, {row}:", f"{row}, {col}:", 1)
            new_lines.append(new_line)
        else:
            new_lines.append(line)
            
    return '\n'.join(new_lines)


# =============================================================================
# Main Test
# =============================================================================

def run_test(num_envs: int, num_steps: int, model: str, verbose: bool = True):
    """Run online RL smoke test with vLLM policy."""
    
    print("=" * 60)
    print("Online RL Smoke Test with vLLM")
    print("=" * 60)
    print(f"Environments: {num_envs}")
    print(f"Steps: {num_steps}")
    print(f"Model: {model}")
    print("=" * 60)
    
    from craftax.craftax_env import make_craftax_env_from_name
    from vllm_policy import VLLMPolicy
    import jax
    
    # Initialize policy
    print("\n[1/3] Loading vLLM policy...")
    policy = VLLMPolicy(model_id=model)
    
    # Initialize environments
    print("\n[2/3] Initializing Craftax environments...")
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    
    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, num_envs)
    
    states = []
    for i, r in enumerate(rngs):
        _, state = env.reset(r, env_params)
        states.append(state)
    
    print(f"  Initialized {num_envs} environments")
    
    # Run RL loop
    print(f"\n[3/3] Running {num_steps} steps...")
    print("-" * 60)
    
    total_reward = 0.0
    episode_count = 0
    step_times = []
    
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        
        # Render and filter observations (matching llm_play_harnessed.py)
        observations = []
        for state in states:
            raw_text = render_craftax_text_swapped(state)
            filtered_text = filter_text_obs(raw_text)
            observations.append(filtered_text)
        
        # Get actions from policy
        actions = policy.get_actions(observations)
        
        # Step environments
        new_states = []
        step_reward = 0.0
        
        for i, (state, action) in enumerate(zip(states, actions)):
            rng, step_rng = jax.random.split(rngs[i])
            rngs = rngs.at[i].set(rng)
            
            _, new_state, reward, done, _ = env.step(
                step_rng, state, int(action), env_params
            )
            
            step_reward += float(reward)
            if done:
                episode_count += 1
            
            new_states.append(new_state)
        
        states = new_states
        total_reward += step_reward
        step_times.append(time.perf_counter() - step_start)
        
        if verbose and (step + 1) % 10 == 0:
            avg_time = np.mean(step_times[-10:])
            sps = num_envs / avg_time
            print(f"  Step {step+1:4d}/{num_steps} | "
                  f"Reward: {step_reward:+5.1f} | "
                  f"Episodes: {episode_count:3d} | "
                  f"Step: {avg_time:.2f}s | "
                  f"SPS: {sps:.1f}")
    
    elapsed = time.perf_counter() - start_time
    
    # Results
    print("-" * 60)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    total_samples = num_envs * num_steps
    samples_per_sec = total_samples / elapsed
    
    print(f"\nPerformance:")
    print(f"  Total samples:    {total_samples:,}")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Samples/sec:      {samples_per_sec:.2f}")
    print(f"  Avg step time:    {np.mean(step_times):.2f}s")
    
    print(f"\nGame stats:")
    print(f"  Episodes done:    {episode_count}")
    print(f"  Total reward:     {total_reward:.1f}")
    
    metrics = policy.get_metrics()
    print(f"\nPolicy stats:")
    print(f"  Parse failures:   {metrics['parse_failures']} ({metrics['failure_rate']*100:.1f}%)")
    
    # Training time estimates
    print("\n" + "=" * 60)
    print("TRAINING TIME ESTIMATES")
    print("=" * 60)
    
    targets = [
        (1e6, "1M"),
        (3e6, "3M"),
        (5e6, "5M"),
        (10e6, "10M"),
        (50e6, "50M"),
    ]
    
    print(f"\nAt {samples_per_sec:.1f} samples/sec:")
    print()
    for target, label in targets:
        hours = target / samples_per_sec / 3600
        days = hours / 24
        status = "✅" if days <= 4 else "⚠️" if days <= 7 else "❌"
        print(f"  {label:>4} steps: {hours:7.1f} hours ({days:5.2f} days) {status}")
    
    max_4days = samples_per_sec * 4 * 24 * 3600
    print(f"\n  Max in 4 days: {max_4days/1e6:.1f}M steps")
    
    return {
        "samples_per_sec": samples_per_sec,
        "total_time": elapsed,
        "episode_count": episode_count,
        "max_4day_steps": max_4days,
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM Online RL smoke test")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    results = run_test(
        num_envs=args.envs,
        num_steps=args.steps,
        model=args.model,
        verbose=not args.quiet,
    )
    
    print("\n✅ Test complete!")
    print(f"\nKey metric: {results['samples_per_sec']:.1f} samples/sec")
    print(f"Feasible in 4 days: {results['max_4day_steps']/1e6:.1f}M steps")


if __name__ == "__main__":
    main()
