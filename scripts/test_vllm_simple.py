#!/usr/bin/env python3
"""
Simple test for vLLM hidden states extraction.
Assumes server is already running.

Usage:
    # Terminal 1: Start server
    bash scripts/start_vllm_hidden.sh --mode last_token

    # Terminal 2: Run this test
    python scripts/test_vllm_simple.py
"""

import sys
import time
import requests
import numpy as np

# Add project root to path
sys.path.insert(0, '.')

def wait_for_server(url="http://localhost:8000", timeout=300):
    """Wait for server to be ready (up to 5 minutes)"""
    print(f"Waiting for vLLM server at {url}...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                print(f"✅ Server ready after {time.time() - start:.1f} seconds")
                return True
        except:
            pass
        time.sleep(2)
        if int(time.time() - start) % 10 == 0:
            print(f"  Still waiting... ({int(time.time() - start)} seconds)")

    print(f"❌ Server not ready after {timeout} seconds")
    return False

def test_extraction():
    """Test hidden state extraction via vLLM"""
    from utils.llm_extractor import VLLMHiddenStateExtractor

    print("\n=== Testing VLLMHiddenStateExtractor ===")

    # Create extractor with 4-layer config (uses last layer)
    print("Creating extractor...")
    extractor = VLLMHiddenStateExtractor(
        server_url="http://localhost:8000",
        model_name="./configs/vllm_hidden_qwen4b",  # 4-layer config
        target_layer=-1,  # Use last of the 4 extracted layers (layer 35)
    )

    # Test observations
    observations = [
        "Player at position 0,0 sees a tree at 1,0",
        "Health is 9.0, Food is 8, Energy is 7",
        "Inventory: Wood: 3, Stone: 5",
    ]

    print(f"\nTesting with {len(observations)} observations...")
    start = time.time()

    hidden_states, metrics = extractor.extract_hidden_states_no_cot(observations)

    elapsed = time.time() - start

    print(f"\n✅ Extraction successful!")
    print(f"  Time: {elapsed:.3f} seconds")
    print(f"  Throughput: {len(observations)/elapsed:.1f} samples/sec")
    print(f"  Hidden shape: {hidden_states.shape}")
    print(f"  Hidden stats: mean={hidden_states.mean():.4f}, std={hidden_states.std():.4f}")
    print(f"  Hidden range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")

    # Test with single-layer config if it exists
    import os
    if os.path.exists("configs/vllm_hidden_last/config.json"):
        print("\n=== Testing Single-Layer Config ===")
        extractor_single = VLLMHiddenStateExtractor(
            server_url="http://localhost:8000",
            model_name="./configs/vllm_hidden_last",  # Single layer (35)
            target_layer=0,  # Only one layer extracted, so index 0
        )

        start = time.time()
        hidden_single, _ = extractor_single.extract_hidden_states_no_cot(observations)
        elapsed_single = time.time() - start

        print(f"✅ Single-layer extraction successful!")
        print(f"  Time: {elapsed_single:.3f} seconds")
        print(f"  Throughput: {len(observations)/elapsed_single:.1f} samples/sec")
        print(f"  Speedup vs 4-layer: {elapsed/elapsed_single:.1f}x")

    return True

def test_raw_api():
    """Test raw vLLM API"""
    print("\n=== Testing Raw vLLM API ===")

    prompt = "The player is at position 0,0 and sees a tree"

    resp = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "./configs/vllm_hidden_qwen4b",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
        },
        timeout=30,
    )

    if resp.status_code == 200:
        result = resp.json()
        print(f"✅ API call successful")

        # Check for hidden states in response
        kv_params = None
        for choice in result.get("choices", []):
            if "kv_transfer_params" in choice:
                kv_params = choice["kv_transfer_params"]
                break

        if not kv_params:
            kv_params = result.get("kv_transfer_params", {})

        if kv_params and "hidden_states_path" in kv_params:
            print(f"  Hidden states saved to: {kv_params['hidden_states_path']}")

            # Try to load and check the file
            import os
            if os.path.exists(kv_params['hidden_states_path']):
                import safetensors.torch
                data = safetensors.torch.load_file(kv_params['hidden_states_path'])
                hs = data["hidden_states"]
                print(f"  Hidden shape: {hs.shape}")
                print(f"  Hidden dtype: {hs.dtype}")
        else:
            print("  ⚠️ No hidden states in response")
    else:
        print(f"❌ API call failed: {resp.status_code}")
        print(resp.text)

def main():
    # Wait for server
    if not wait_for_server():
        print("\n⚠️ Server not ready. Please start it manually:")
        print("  bash scripts/start_vllm_hidden.sh --mode last_token")
        return 1

    # Test raw API
    try:
        test_raw_api()
    except Exception as e:
        print(f"❌ Raw API test failed: {e}")

    # Test extractor
    try:
        test_extraction()
    except Exception as e:
        print(f"❌ Extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n✨ All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())