#!/usr/bin/env python3
"""
Quick profiling without JAX compilation overhead.
Uses pre-rendered text to measure just the Python overhead.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import requests

def profile_text_filtering():
    """Profile just the text filtering part."""
    from utils.llm_prompts import filter_text_obs

    # Sample Craftax text observation
    sample_text = """Row 0, Col 0: Grass
Row 0, Col 1: Stone
Row 0, Col 2: Tree
Row 1, Col 0: Grass
Row 1, Col 1: Player
Row 1, Col 2: Grass
Row 2, Col 0: Stone
Row 2, Col 1: Grass
Row 2, Col 2: Water
Inventory:
  Wood: 0
  Stone: 0
  Iron: 0
  Diamond: 0
  Sapling: 0
Status:
  Health: 10
  Hunger: 10
  Thirst: 10
  Energy: 10"""

    print("Profiling text filtering...")
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        filtered = filter_text_obs(sample_text)
        times.append(time.perf_counter() - t0)

    avg_time = np.mean(times) * 1000  # ms
    print(f"  Average time: {avg_time:.3f} ms")
    print(f"  Min/Max: {np.min(times)*1000:.3f}/{np.max(times)*1000:.3f} ms")

    return avg_time


def profile_vllm_latency():
    """Profile vLLM server round-trip latency."""

    # Check server
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code != 200:
            print("❌ vLLM server not running")
            return None
    except:
        print("❌ vLLM server not reachable")
        return None

    print("\nProfiling vLLM latency...")

    from utils.llm_extractor import VLLMHiddenStateExtractor

    model_name = "./configs/vllm_hidden_qwen4b"
    model_id = "Qwen/Qwen3-4B-Thinking-2507"

    extractor = VLLMHiddenStateExtractor(
        server_url="http://localhost:8000",
        model_name=model_name,
        model_id=model_id,
        target_layer=-1,
    )

    test_obs = ["Row 5, Col 5: Player\nInventory: Wood: 2"]

    # Test different batch sizes
    batch_tests = [1, 8, 16, 32, 64, 128]

    print("\nBatch | Total(ms) | Per-env(ms) | Envs/sec")
    print("-" * 45)

    results = {}
    for batch in batch_tests:
        obs_batch = test_obs * batch

        # Warmup
        _ = extractor.extract_hidden_states_no_cot(obs_batch, batch_size=min(32, batch))

        # Measure
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _, _ = extractor.extract_hidden_states_no_cot(obs_batch, batch_size=min(32, batch))
            times.append(time.perf_counter() - t0)

        avg_time = np.mean(times)
        per_env = avg_time / batch
        envs_per_sec = batch / avg_time

        print(f"{batch:5d} | {avg_time*1000:9.1f} | {per_env*1000:11.2f} | {envs_per_sec:8.1f}")
        results[batch] = {
            'total_ms': avg_time * 1000,
            'per_env_ms': per_env * 1000,
            'throughput': envs_per_sec
        }

    return results


def estimate_total_performance():
    """Estimate total performance based on measurements."""

    print("\n" + "=" * 60)
    print("PERFORMANCE ESTIMATION")
    print("=" * 60)

    # Baseline PPO (no text, no LLM)
    baseline_sps = 18500
    ms_per_step_baseline = 128 / baseline_sps * 1000

    print(f"\nBaseline PPO (128 envs):")
    print(f"  {baseline_sps:,} SPS = {ms_per_step_baseline:.2f} ms/step")

    # Text filtering overhead (assume ~0.1 ms per env based on typical Python perf)
    text_filter_ms = 0.1 * 128

    # LLM inference (from your observed ~5 SPS with skip=1)
    # 5 SPS with 128 envs = 200ms per step total
    llm_ms = 200 - ms_per_step_baseline - text_filter_ms

    print(f"\nEstimated component times (128 envs):")
    print(f"  JAX env step:    {ms_per_step_baseline:.1f} ms ({ms_per_step_baseline/200*100:.1f}%)")
    print(f"  Text filtering:  {text_filter_ms:.1f} ms ({text_filter_ms/200*100:.1f}%)")
    print(f"  LLM inference:   {llm_ms:.1f} ms ({llm_ms/200*100:.1f}%)")
    print(f"  TOTAL:           200.0 ms (5 SPS)")

    print(f"\nWith skip_n:")
    for skip_n in [5, 25, 50, 100]:
        # Only pay text+LLM cost every skip_n steps
        avg_ms = (ms_per_step_baseline * (skip_n - 1) + 200) / skip_n
        sps = 1000 / avg_ms * 128
        print(f"  skip_n={skip_n:3d}: {sps:6.0f} SPS ({avg_ms:.1f} ms/step avg)")


if __name__ == "__main__":
    print("=" * 60)
    print("Quick Performance Profiling")
    print("=" * 60)
    print()

    # Test 1: Text filtering
    filter_time = profile_text_filtering()

    # Test 2: vLLM latency
    vllm_results = profile_vllm_latency()

    # Test 3: Estimate total
    estimate_total_performance()

    if vllm_results:
        print("\n" + "=" * 60)
        print("ACTUAL MEASUREMENTS")
        print("=" * 60)

        if 128 in vllm_results:
            llm_128 = vllm_results[128]['total_ms']
            text_128 = filter_time * 128
            env_step = 128 / 18500 * 1000  # From baseline

            total = env_step + text_128 + llm_128

            print(f"\nMeasured for 128 environments:")
            print(f"  Env step:       {env_step:.1f} ms ({env_step/total*100:.1f}%)")
            print(f"  Text filter:    {text_128:.1f} ms ({text_128/total*100:.1f}%)")
            print(f"  LLM inference:  {llm_128:.1f} ms ({llm_128/total*100:.1f}%)")
            print(f"  TOTAL:          {total:.1f} ms")
            print(f"  Expected SPS:   {1000/total*128:.1f}")
    else:
        print("\nStart vLLM server for complete measurements:")
        print("  bash scripts/start_vllm_hidden.sh --mode last_token")