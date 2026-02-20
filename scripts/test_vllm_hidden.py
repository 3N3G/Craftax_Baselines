#!/usr/bin/env python3
"""
Test script for vLLM Hidden States Extractor Plugin.

Usage:
  # 1. Start the server (in a separate terminal):
  bash scripts/start_vllm_hidden.sh

  # 2. Run this test:
  python scripts/test_vllm_hidden.py

  # Or with generation:
  python scripts/test_vllm_hidden.py --max-tokens 50
"""

import argparse
import json
import os
import time
import glob

import requests
import safetensors.torch


def test_hidden_states(
    server_url: str = "http://localhost:8000",
    model: str = "./configs/vllm_hidden_qwen4b",
    max_tokens: int = 1,
    hidden_states_path: str = "/tmp/hidden_states",
    prompt: str = "What is 2+2?",
):
    print(f"=== vLLM Hidden States Test ===")
    print(f"Server: {server_url}")
    print(f"Model: {model}")
    print(f"Max tokens: {max_tokens}")
    print(f"Hidden states path: {hidden_states_path}")
    print()

    # Clean up old hidden states
    for f in glob.glob(os.path.join(hidden_states_path, "*.safetensors")):
        os.remove(f)

    # Send request
    print(f"Sending request: '{prompt[:50]}...'")
    t0 = time.perf_counter()
    resp = requests.post(
        f"{server_url}/v1/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
    )
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code}")
        print(resp.text)
        return

    result = resp.json()
    print(f"Response received in {elapsed:.3f}s")

    # Print response
    if result.get("choices"):
        text = result["choices"][0].get("text", "")
        print(f"Generated text: '{text[:100]}...'")

    # Check for kv_transfer_params
    kv_params = None
    for choice in result.get("choices", []):
        if "kv_transfer_params" in choice:
            kv_params = choice["kv_transfer_params"]
    
    # Also check top-level
    if kv_params is None:
        kv_params = result.get("kv_transfer_params")

    if kv_params:
        print(f"\nKV Transfer Params: {json.dumps(kv_params, indent=2)}")
        hs_path = kv_params.get("hidden_states_path")
    else:
        print("\nNo kv_transfer_params in response, scanning directory...")
        files = sorted(glob.glob(os.path.join(hidden_states_path, "*.safetensors")))
        if files:
            hs_path = files[-1]
            print(f"Found: {hs_path}")
        else:
            print("ERROR: No hidden states files found!")
            return

    # Load and verify hidden states
    if hs_path and os.path.exists(hs_path):
        print(f"\nLoading hidden states from: {hs_path}")
        data = safetensors.torch.load_file(hs_path)

        hs = data["hidden_states"]
        tids = data["token_ids"]

        print(f"  hidden_states shape: {hs.shape}")
        print(f"  hidden_states dtype: {hs.dtype}")
        print(f"  token_ids shape:     {tids.shape}")
        print(f"  hidden_states range: [{hs.min():.4f}, {hs.max():.4f}]")
        print(f"  hidden_states mean:  {hs.mean():.4f}")
        print(f"  hidden_states std:   {hs.std():.4f}")
        
        # Check norms per layer
        for i in range(hs.shape[0]):
            layer_hs = hs[i]
            norm = layer_hs.float().norm(dim=-1).mean()
            print(f"  Layer {i} mean norm: {norm:.4f}")

        print("\n✅ Hidden states extraction working!")
    else:
        print(f"\nERROR: Hidden states file not found at: {hs_path}")


def test_batch(
    server_url: str = "http://localhost:8000",
    model: str = "./configs/vllm_hidden_qwen4b",
    num_requests: int = 8,
    max_tokens: int = 1,
    hidden_states_path: str = "/tmp/hidden_states",
):
    """Test with multiple concurrent requests."""
    import concurrent.futures

    prompts = [
        "What is 2+2?",
        "Tell me about cats.",
        "Write a haiku about rain.",
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "What color is the sky?",
        "Count to five.",
        "Name three fruits.",
    ]

    # Clean up
    for f in glob.glob(os.path.join(hidden_states_path, "*.safetensors")):
        os.remove(f)

    print(f"\n=== Batch Test: {num_requests} requests ===")
    t0 = time.perf_counter()

    def send_one(prompt):
        resp = requests.post(
            f"{server_url}/v1/completions",
            headers={"Content-Type": "application/json"},
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens},
        )
        return resp.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as pool:
        results = list(pool.map(send_one, prompts[:num_requests]))

    elapsed = time.perf_counter() - t0
    print(f"All {num_requests} requests completed in {elapsed:.3f}s")
    print(f"Throughput: {num_requests / elapsed:.1f} requests/sec")

    # Check files
    files = sorted(glob.glob(os.path.join(hidden_states_path, "*.safetensors")))
    print(f"Hidden state files written: {len(files)}")

    if files:
        data = safetensors.torch.load_file(files[0])
        print(f"Sample shape: {data['hidden_states'].shape}")

    print("✅ Batch test complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default="./configs/vllm_hidden_qwen4b")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--hidden-states-path", default="/tmp/hidden_states")
    parser.add_argument("--batch", action="store_true", help="Run batch test")
    parser.add_argument("--num-requests", type=int, default=8)
    args = parser.parse_args()

    test_hidden_states(
        server_url=args.server_url,
        model=args.model,
        max_tokens=args.max_tokens,
        hidden_states_path=args.hidden_states_path,
    )

    if args.batch:
        test_batch(
            server_url=args.server_url,
            model=args.model,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            hidden_states_path=args.hidden_states_path,
        )
