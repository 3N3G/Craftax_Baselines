#!/usr/bin/env python3
"""
Benchmark: vLLM Hidden States Plugin vs HuggingFace (No-CoT Mode)

Compares hidden state extraction throughput for prompt-only forward passes:
1. HuggingFace: extract_hidden_states_no_cot (last-token hidden from last layer)
2. vLLM Plugin: server mode with last_token connector (last-token hidden from 4 layers)

Usage:
  # Start vLLM server first (Terminal 1):
  vllm serve ./configs/vllm_hidden_qwen4b \
      --max-model-len 8192 --gpu-memory-utilization 0.95 \
      --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/tmp/hidden_states","mode":"last_token"}}'

  # Run benchmark (Terminal 2):
  python scripts/benchmark_hidden_no_cot.py --backend vllm
  python scripts/benchmark_hidden_no_cot.py --backend huggingface
  python scripts/benchmark_hidden_no_cot.py --backend both
"""

import argparse
import concurrent.futures
import glob
import os
import sys
import torch
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Add project root to path so `from utils.xxx import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.llm_prompts import create_prompt

def create_tiny_prompt(obs_text: str, tokenizer) -> str:
    """Minimal prompt for fast calibration without the 2.8k system prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": obs_text}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =============================================================================
# Realistic Craftax prompts (varying lengths)
# =============================================================================
SAMPLE_OBSERVATIONS = [
    "Craftax is a game about surviving.",
    "Map (interesting tiles only): 1, 0:crafting_table, 2, -3:tree, 4, 0:stone\nInventory:\nWood: 3\nHealth: 9.0\nFood: 9\nDrink: 9\nEnergy: 9\nDirection: right",
    "Map (interesting tiles only): 0, -2:water, 3, -4:tree, -3, -3:tree, 4, -3:Skeleton on grass\nInventory:\nHealth: 4.0\nFood: 5\nDrink: 4\nDirection: down",
    "Map (interesting tiles only): 0, 2:Arrow, 0, 3:Skeleton, 1, 1:Cow\nInventory:\nWood: 0\nHealth: 3.0\nDirection: right",
    "Map (interesting tiles only): 1, 0:Orc Soldier on torch on path, 4, 0:Snail on path\nInventory:\nWood: 17\nIron: 4\nStone: 61\nIron Sword with No enchantment\nIron Helmet with No enchantment\nHealth: 6.3\nFood: 3\nDirection: down",
    "Map: [No interesting tiles in view]\nInventory: Wood:10, Stone:5\nHealth:9\nDirection:left",
    "Map (interesting tiles only): -2, 1:diamond, 0, -1:lava, 1, 2:zombie\nInventory:\nWood: 5\nStone: 12\nIron: 3\nStone Pickaxe\nIron Sword\nHealth: 7.5\nFood: 6\nDrink: 8\nEnergy: 4\nDirection: up",
    "Map (interesting tiles only): -1, 0:ladder_down (open), 2, 3:bat, -3, 1:iron\nInventory:\nWood: 8\nStone: 20\nCoal: 5\nIron: 2\nIron Pickaxe\nIron Sword\nIron Helmet\nHealth: 8.0\nFood: 7\nDrink: 5\nEnergy: 6\nDirection: up\nFloor: 2\nKills: 8/8",
    "Map (interesting tiles only): 0, 1:tree, 0, -1:tree, 1, 0:stone, -1, 0:cow\nInventory:\nWood: 0\nHealth: 9.0\nFood: 9\nDrink: 9\nEnergy: 9\nDirection: right\nFloor: 1",
]


def get_prompts(n: int) -> List[str]:
    """Generate n prompts by cycling through samples."""
    return [SAMPLE_OBSERVATIONS[i % len(SAMPLE_OBSERVATIONS)] for i in range(n)]


# =============================================================================
# HuggingFace Backend
# =============================================================================

def benchmark_huggingface(
    model_id: str,
    prompts: List[str],
    batch_size: int,
    num_warmup: int = 2,
) -> Dict[str, Any]:
    """Benchmark HuggingFace extract_hidden_states_no_cot."""
    from utils.llm_extractor import LLMHiddenStateExtractor

    print(f"\n{'='*60}")
    print(f"HuggingFace Benchmark")
    print(f"  Model: {model_id}")
    print(f"  Prompts: {len(prompts)}, Batch size: {batch_size}")
    print(f"{'='*60}")

    extractor = LLMHiddenStateExtractor(model_id=model_id)

    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Warmup (one by one if Flash Attention is off)
    warmup_batch = 1 if not extractor.flash_enabled else batch_size
    print(f"Warming up ({num_warmup} samples, batch_size={warmup_batch})...")
    warmup_prompts = prompts[:num_warmup]
    extractor.extract_hidden_states_no_cot(warmup_prompts, batch_size=warmup_batch)
    
    extractor.total_samples = 0
    extractor.total_time = 0.0
    extractor.batch_times = []

    # Layer Scan (just one prompt to avoid OOM)
    # We use a tiny prompt for the scan (no massive Craftax system prompt)
    print("Scanning layer norms for one TINY prompt (calibration)...")
    with torch.no_grad():
        tiny_prompt = create_tiny_prompt(prompts[0], extractor.tokenizer)
        inputs = extractor.tokenizer(tiny_prompt, return_tensors="pt").to(extractor.model.device)
        
        print(f"  Tiny prompt: {tiny_prompt}")
        print(f"  Sequence length (tokens): {inputs['input_ids'].shape[1]}")
        
        outputs = extractor.model(**inputs, output_hidden_states=True)
        all_layer_states = outputs.hidden_states
        print(f"  Number of hidden state entries in HF: {len(all_layer_states)}")
    
    # stats for each layer
    print("\nHuggingFace Layer Norms (Last Token):")
    # Use the one we previously selected ([-2]) for the main result
    hidden_states_for_norm_scan = all_layer_states[-2][:, -1, :].to(torch.float32).cpu().numpy()

    for i, layer_hs in enumerate(all_layer_states):
        # (1, seq, hidden) -> (hidden,)
        last_token_hs = layer_hs[0, -1, :].to(torch.float32).cpu().numpy()
        norm = float(np.linalg.norm(last_token_hs))
        mean = float(np.mean(last_token_hs))
        std = float(np.std(last_token_hs))
        
        suffix = ""
        if i == 0: suffix = " (Embeddings)"
        if i == len(all_layer_states) - 2: suffix = " (Output Layer 35 / Pre-Norm)"
        if i == len(all_layer_states) - 1: suffix = " (Post-Norm)"
        print(f"  Layer {i:2d}: Norm={norm:6.1f} | Mean={mean:7.4f} | Std={std:7.4f}{suffix}")

    # Benchmark correctly batched
    print("\nRunning actual benchmark...")
    t0 = time.perf_counter()
    hidden_states, _ = extractor.extract_hidden_states_no_cot(
        prompts, batch_size=batch_size
    )
    elapsed = time.perf_counter() - t0

    result = {
        "backend": "huggingface",
        "num_prompts": len(prompts),
        "batch_size": batch_size,
        "total_time_s": elapsed,
        "samples_per_sec": len(prompts) / elapsed,
        "mean_latency_s": elapsed / len(prompts),
        "hidden_shape": list(hidden_states.shape),
        "hidden_mean": float(np.mean(hidden_states)),
        "hidden_std": float(np.std(hidden_states)),
        "hidden_norm_mean": float(np.mean(np.linalg.norm(hidden_states, axis=1))),
    }

    # Print first 5 elements for comparison
    first_hs = hidden_states[0, :5]
    print(f"  First 5 elements: {first_hs}")
    print(f"  Hidden Mean: {result['hidden_mean']:.4f}")
    print(f"  Hidden Norm Mean: {result['hidden_norm_mean']:.1f}")

    return result


# =============================================================================
# vLLM Backend
# =============================================================================

def benchmark_vllm(
    server_url: str,
    model_name: str,
    prompts: List[str],
    batch_size: int,
    hidden_states_path: str,
    num_warmup: int = 2,
    target_layer: int = -1,
) -> Dict[str, Any]:
    """
    Benchmark vLLM hidden states plugin.
    
    Sends concurrent requests, reads safetensors files.
    target_layer: which layer to use from the 4 extracted (-1 = last layer).
    """
    import requests as req_lib
    import safetensors.torch

    print(f"\n{'='*60}")
    print(f"vLLM Plugin Benchmark")
    print(f"  Server: {server_url}")
    print(f"  Prompts: {len(prompts)}, Batch size (concurrency): {batch_size}")
    print(f"{'='*60}")

    def send_request(prompt: str) -> dict:
        resp = req_lib.post(
            f"{server_url}/v1/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0.0,
            },
        )
        return resp.json()

    def get_hidden_state(result: dict) -> np.ndarray:
        """Extract hidden state from response + safetensors file."""
        kv_params = None
        for choice in result.get("choices", []):
            if "kv_transfer_params" in choice:
                kv_params = choice["kv_transfer_params"]
        if kv_params is None:
            kv_params = result.get("kv_transfer_params", {})

        hs_path = kv_params.get("hidden_states_path")
        if not hs_path or not os.path.exists(hs_path):
            # Fallback: scan directory
            files = sorted(glob.glob(os.path.join(hidden_states_path, "*.safetensors")))
            if files:
                hs_path = files[-1]
            else:
                return None

        data = safetensors.torch.load_file(hs_path)
        hs = data["hidden_states"]  # [num_layers, 1, hidden_size]
        # Use target layer, squeeze the seq dim
        return hs[target_layer, 0, :].float().numpy()

    # Clean old files
    for f in glob.glob(os.path.join(hidden_states_path, "*.safetensors")):
        os.remove(f)

    # Warmup
    print(f"Warming up ({num_warmup} samples)...")
    for p in prompts[:num_warmup]:
        r = send_request(p)
        get_hidden_state(r)

    # Clean again
    for f in glob.glob(os.path.join(hidden_states_path, "*.safetensors")):
        os.remove(f)

    # Benchmark: send in batches of `batch_size` concurrently
    print("Running benchmark...")
    all_hidden = []
    batch_latencies = []

    t0 = time.perf_counter()
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        batch_t0 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_prompts)) as pool:
            results = list(pool.map(send_request, batch_prompts))

        # Read hidden states
        for r in results:
            hs = get_hidden_state(r)
            if hs is not None:
                all_hidden.append(hs)

        batch_latencies.append(time.perf_counter() - batch_t0)

    elapsed = time.perf_counter() - t0

    if all_hidden:
        hidden_states = np.stack(all_hidden)
    else:
        hidden_states = np.array([])

    result = {
        "backend": "vllm_plugin",
        "num_prompts": len(prompts),
        "batch_size": batch_size,
        "total_time_s": elapsed,
        "samples_per_sec": len(prompts) / elapsed,
        "mean_latency_s": elapsed / len(prompts),
        "batch_latency_mean_s": float(np.mean(batch_latencies)),
        "batch_latency_std_s": float(np.std(batch_latencies)),
        "hidden_shape": list(hidden_states.shape) if len(hidden_states) > 0 else [],
        "hidden_mean": float(np.mean(hidden_states)) if len(hidden_states) > 0 else 0,
        "hidden_std": float(np.std(hidden_states)) if len(hidden_states) > 0 else 0,
        "hidden_norm_mean": float(np.mean(np.linalg.norm(hidden_states, axis=1))) if len(hidden_states) > 0 else 0,
        "target_layer": target_layer,
    }

    # Print first 5 elements for comparison
    first_hs = hidden_states[0, :5] if len(hidden_states) > 0 else []
    print(f"  First 5 elements: {first_hs}")
    print(f"  Hidden Mean: {result['hidden_mean']:.4f}")
    print(f"  Hidden Norm Mean: {result['hidden_norm_mean']:.1f}")

    return result


# =============================================================================
# Results
# =============================================================================

def print_comparison(results: List[Dict[str, Any]]):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"{'COMPARISON':^70}")
    print(f"{'='*70}")

    headers = ["Metric"] + [r["backend"] for r in results]
    rows = [
        ("Samples", [str(r["num_prompts"]) for r in results]),
        ("Batch Size", [str(r["batch_size"]) for r in results]),
        ("Total Time (s)", [f"{r['total_time_s']:.3f}" for r in results]),
        ("Throughput (samples/s)", [f"{r['samples_per_sec']:.1f}" for r in results]),
        ("Latency (ms/sample)", [f"{r['mean_latency_s']*1000:.1f}" for r in results]),
        ("Hidden Shape", [str(r.get('hidden_shape', 'N/A')) for r in results]),
        ("Hidden Mean", [f"{r.get('hidden_mean', 0):.4f}" for r in results]),
        ("Hidden Std", [f"{r.get('hidden_std', 0):.4f}" for r in results]),
        ("Hidden Norm Mean", [f"{r.get('hidden_norm_mean', 0):.1f}" for r in results]),
    ]

    col_widths = [max(len(h), max(len(v) for v in vals) if vals else 0) + 2
                  for h, vals in [(headers[0], [r[0] for r in rows])] +
                  [(headers[i+1], [r[1][i] for r in rows]) for i in range(len(results))]]

    # Print header
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for label, vals in rows:
        row_vals = [label] + vals
        print("".join(v.ljust(w) for v, w in zip(row_vals, col_widths)))

    # Speedup
    if len(results) == 2:
        speedup = results[0]["mean_latency_s"] / max(results[1]["mean_latency_s"], 1e-9)
        print(f"\n  Speedup: {speedup:.1f}x " +
              f"({results[1]['backend']} vs {results[0]['backend']})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark hidden state extraction")
    parser.add_argument("--backend", choices=["huggingface", "vllm", "both"],
                        default="both")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="./configs/vllm_hidden_qwen4b")
    parser.add_argument("--hidden-states-path", default="/tmp/hidden_states")
    parser.add_argument("--target-layer", type=int, default=-1,
                        help="Which layer to use from vLLM (0-3, -1=last)")
    args = parser.parse_args()

    prompts = get_prompts(args.num_prompts)
    results = []

    if args.backend in ("huggingface", "both"):
        r = benchmark_huggingface(
            model_id=args.model_id,
            prompts=prompts,
            batch_size=args.batch_size,
        )
        results.append(r)

    if args.backend in ("vllm", "both"):
        r = benchmark_vllm(
            server_url=args.server_url,
            model_name=args.model_name,
            prompts=prompts,
            batch_size=args.batch_size,
            hidden_states_path=args.hidden_states_path,
            target_layer=args.target_layer,
        )
        results.append(r)

    if results:
        print_comparison(results)


if __name__ == "__main__":
    main()
