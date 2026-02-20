#!/usr/bin/env python3
"""
Test batched vLLM inference for 128 environments.

Compares:
1. Current approach: concurrent individual requests
2. Optimized approach: single batched request
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from utils.llm_extractor import VLLMHiddenStateExtractor
from utils.vllm_batch_extractor import VLLMBatchExtractor


def test_batch_sizes():
    """Test different batch sizes to find optimal configuration."""

    # Generate test observations
    test_observations = [
        f"Row 5, Col 5: Player\nRow 4, Col 5: Grass\nInventory: Wood: 2, Stone: 1"
        for _ in range(128)
    ]

    print("=" * 60)
    print("vLLM Batch Performance Test")
    print("=" * 60)

    # Test 1: Current approach (8 environments, for comparison)
    print("\n1. Current approach (8 concurrent requests):")
    extractor_current = VLLMHiddenStateExtractor(
        max_workers=8
    )

    small_obs = test_observations[:8]
    start = time.perf_counter()
    hidden, _ = extractor_current.extract_hidden_states_no_cot(small_obs, batch_size=8)
    elapsed = time.perf_counter() - start
    print(f"   Time for 8 envs: {elapsed:.2f}s")
    print(f"   Throughput: {8/elapsed:.1f} envs/sec")
    print(f"   Projected for 128: {128/(8/elapsed):.1f}s")

    # Test 2: Current approach with more workers (128 environments)
    print("\n2. Current approach scaled (32 concurrent requests):")
    extractor_scaled = VLLMHiddenStateExtractor(
        max_workers=32
    )

    start = time.perf_counter()
    hidden, _ = extractor_scaled.extract_hidden_states_no_cot(test_observations, batch_size=32)
    elapsed = time.perf_counter() - start
    print(f"   Time for 128 envs: {elapsed:.2f}s")
    print(f"   Throughput: {128/elapsed:.1f} envs/sec")

    # Test 3: Optimized batched approach
    print("\n3. Optimized batch approach (single request):")
    extractor_batch = VLLMBatchExtractor()

    start = time.perf_counter()
    hidden, metrics = extractor_batch.extract_batch(test_observations)
    elapsed = time.perf_counter() - start
    print(f"   Time for 128 envs: {elapsed:.2f}s")
    print(f"   Throughput: {128/elapsed:.1f} envs/sec")
    print(f"   Speedup vs current: {(128/(8/elapsed))/elapsed:.1f}x")

    # Verify output shapes
    print(f"\n4. Output verification:")
    print(f"   Hidden shape: {hidden.shape}")
    print(f"   Expected: (128, 2560)")
    print(f"   Match: {hidden.shape == (128, 2560)}")

    return hidden, metrics


if __name__ == "__main__":
    print("Starting vLLM batch test...")
    print("Make sure vLLM server is running with:")
    print("  bash scripts/start_vllm_hidden.sh --mode last_token\n")

    try:
        hidden, metrics = test_batch_sizes()
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()