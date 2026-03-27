#!/usr/bin/env python3
"""
Profile online RL with LLM hidden states to identify bottlenecks.

Measures time for each component:
1. Text rendering (render_craftax_text_swapped + filter_text_obs)
2. LLM inference (HTTP request + hidden state extraction)
3. Policy forward pass
4. Environment stepping
5. Other overhead
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import jax
from craftax.craftax_env import make_craftax_env_from_name

# Set CPU for JAX to avoid GPU conflicts
os.environ["JAX_PLATFORM_NAME"] = "cpu"

def profile_components(num_envs=128, num_steps=100):
    """Profile each component of the online RL loop."""

    print("=" * 60)
    print(f"Profiling Online RL Components ({num_envs} environments)")
    print("=" * 60)

    # Initialize environments
    print("\n1. Setting up environments...")
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, num_envs)

    states = []
    symbolic_obs_list = []

    for i, r in enumerate(rngs):
        obs, state = env.reset(r, env_params)
        states.append(state)
        symbolic_obs_list.append(np.array(obs))

    print(f"   Initialized {num_envs} environments")
    print(f"   Symbolic obs shape: {symbolic_obs_list[0].shape}")

    # Import rendering functions
    from online_rl_llm.online_rl_hidden import render_craftax_text_swapped
    from utils.llm_prompts import filter_text_obs

    # WARMUP: Pre-compile JAX functions
    print("\n2. Warming up JAX (this may take 1-2 minutes on first run)...")
    print("   Compiling render_craftax_text...")
    _ = render_craftax_text_swapped(states[0])

    print("   Compiling env.step...")
    rng, step_rng = jax.random.split(rngs[0])
    _ = env.step(step_rng, states[0], 0, env_params)

    print("   Warmup complete!")

    # Timing storage
    text_render_times = []
    filter_times = []
    env_step_times = []

    print(f"\n3. Profiling {num_steps} steps...")
    print("-" * 40)

    for step in range(num_steps):
        # Profile text rendering
        t0 = time.perf_counter()
        text_observations = []
        for state in states:
            t_render = time.perf_counter()
            raw_text = render_craftax_text_swapped(state)
            text_render_times.append(time.perf_counter() - t_render)

            t_filter = time.perf_counter()
            filtered_text = filter_text_obs(raw_text)
            filter_times.append(time.perf_counter() - t_filter)

            text_observations.append(filtered_text)
        text_total_time = time.perf_counter() - t0

        # Random actions for stepping
        actions = np.random.randint(0, 43, size=num_envs)

        # Profile environment stepping
        t0 = time.perf_counter()
        new_states = []
        for i, (state, action) in enumerate(zip(states, actions)):
            rng, step_rng = jax.random.split(rngs[i])
            rngs = rngs.at[i].set(rng)

            obs, new_state, reward, done, _ = env.step(
                step_rng, state, int(action), env_params
            )
            new_states.append(new_state)
            symbolic_obs_list[i] = np.array(obs)

        states = new_states
        env_step_times.append(time.perf_counter() - t0)

        if (step + 1) % 20 == 0:
            print(f"   Step {step + 1}/{num_steps}")

    # Calculate statistics
    print("\n4. Results")
    print("=" * 60)

    avg_text_render = np.mean(text_render_times) * 1000  # Convert to ms
    avg_filter = np.mean(filter_times) * 1000
    avg_env_step = np.mean(env_step_times) * 1000

    # Per-step totals
    text_per_step = (avg_text_render + avg_filter) * num_envs

    print(f"\nPER ENVIRONMENT (averaged over {len(text_render_times)} calls):")
    print(f"  Text rendering:     {avg_text_render:.3f} ms")
    print(f"  Text filtering:     {avg_filter:.3f} ms")
    print(f"  Total per env:      {avg_text_render + avg_filter:.3f} ms")

    print(f"\nPER STEP ({num_envs} environments):")
    print(f"  Text processing:    {text_per_step:.1f} ms")
    print(f"  Environment step:   {avg_env_step * 1000:.1f} ms")
    print(f"  Total (no LLM):     {text_per_step + avg_env_step * 1000:.1f} ms")

    print(f"\nTHEORETICAL SPEEDS:")
    no_text_sps = 1000 / (avg_env_step * 1000)
    with_text_sps = 1000 / (text_per_step + avg_env_step * 1000)
    print(f"  Without text rendering: {no_text_sps * num_envs:.0f} SPS")
    print(f"  With text (every step):  {with_text_sps * num_envs:.0f} SPS")
    print(f"  With text (skip_n=25):   {(no_text_sps * 24 + with_text_sps) / 25 * num_envs:.0f} SPS")

    return {
        "text_render_ms": avg_text_render,
        "filter_ms": avg_filter,
        "env_step_ms": avg_env_step * 1000,
        "text_per_step_ms": text_per_step,
    }


def profile_llm_inference():
    """Profile LLM inference separately."""
    print("\n" + "=" * 60)
    print("Profiling LLM Inference")
    print("=" * 60)

    # Check if vLLM server is running
    import requests
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code != 200:
            print("❌ vLLM server not running. Start it with:")
            print("   bash scripts/start_vllm_hidden.sh --mode last_token")
            return None
    except:
        print("❌ vLLM server not reachable. Start it with:")
        print("   bash scripts/start_vllm_hidden.sh --mode last_token")
        return None

    print("✅ vLLM server is running")

    # Test with different batch sizes
    from utils.llm_extractor import VLLMHiddenStateExtractor

    model_name = "./configs/vllm_hidden_qwen4b"
    model_id = "Qwen/Qwen3-4B-Thinking-2507"

    extractor = VLLMHiddenStateExtractor(
        server_url="http://localhost:8000",
        model_name=model_name,
        model_id=model_id,
        target_layer=-1,
    )

    test_obs = [
        "Row 5, Col 5: Player\nRow 4, Col 5: Grass\nInventory: Wood: 2"
    ]

    batch_sizes = [1, 8, 16, 32, 64, 128]

    print("\nBatch Size | Time (ms) | Per-env (ms) | Throughput (envs/sec)")
    print("-" * 65)

    results = {}
    for batch_size in batch_sizes:
        batch_obs = test_obs * batch_size

        # Warm up
        _ = extractor.extract_hidden_states_no_cot(batch_obs, batch_size=min(32, batch_size))

        # Measure
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            hidden, _ = extractor.extract_hidden_states_no_cot(batch_obs, batch_size=min(32, batch_size))
            times.append(time.perf_counter() - t0)

        avg_time = np.mean(times)
        per_env = avg_time / batch_size * 1000
        throughput = batch_size / avg_time

        print(f"{batch_size:10d} | {avg_time*1000:9.1f} | {per_env:12.2f} | {throughput:15.1f}")
        results[batch_size] = avg_time * 1000

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=128, help="Number of environments")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to profile")
    parser.add_argument("--profile-llm", action="store_true", help="Also profile LLM inference")
    args = parser.parse_args()

    # Profile main components
    timings = profile_components(num_envs=args.envs, num_steps=args.steps)

    # Optionally profile LLM
    if args.profile_llm:
        llm_timings = profile_llm_inference()

        if llm_timings:
            print("\n" + "=" * 60)
            print("COMPLETE BREAKDOWN")
            print("=" * 60)

            # Calculate total time per step
            text_time = timings["text_per_step_ms"]
            env_time = timings["env_step_ms"]
            llm_time = llm_timings.get(args.envs, llm_timings.get(128, 1000))

            print(f"\nFor {args.envs} environments:")
            print(f"  Environment step:  {env_time:6.1f} ms ({env_time/(env_time+text_time+llm_time)*100:4.1f}%)")
            print(f"  Text processing:   {text_time:6.1f} ms ({text_time/(env_time+text_time+llm_time)*100:4.1f}%)")
            print(f"  LLM inference:     {llm_time:6.1f} ms ({llm_time/(env_time+text_time+llm_time)*100:4.1f}%)")
            print(f"  TOTAL:             {env_time+text_time+llm_time:6.1f} ms")

            print(f"\nExpected SPS:")
            print(f"  skip_n=1:  {1000/(env_time+text_time+llm_time) * args.envs:6.1f} SPS (all components every step)")
            print(f"  skip_n=5:  {1000/((env_time+text_time+llm_time) + 4*env_time)/5 * args.envs:6.1f} SPS")
            print(f"  skip_n=25: {1000/((env_time+text_time+llm_time) + 24*env_time)/25 * args.envs:6.1f} SPS")

            print(f"\nBaseline PPO: ~18,500 SPS")
    else:
        print("\nTo profile LLM inference, run:")
        print(f"  python scripts/profile_online_rl.py --envs {args.envs} --profile-llm")