#!/usr/bin/env python3
"""
Benchmark: vLLM Hidden States Extractor Plugin vs HuggingFace

Compares:
1. HuggingFace + Flash Attention (current approach in online_rl_hidden.py)
2. vLLM Hidden States Extractor Plugin (fynnsu/vllm-hidden-states-extractor)

The vLLM plugin approach:
- Uses Eagle3 speculative decoding plumbing
- Creates a dummy model that caches hidden states into KV cache
- Extracts via KV connector to disk/memory
- Requires vLLM 0.14.0+ with the plugin installed

Usage:
    # Start vLLM server with plugin first (see setup instructions)
    # Then run:
    python benchmark_vllm_hidden_plugin.py --backend all --samples 16
    python benchmark_vllm_hidden_plugin.py --backend huggingface --samples 16
    python benchmark_vllm_hidden_plugin.py --backend vllm-plugin --samples 16

vLLM Server Setup:
    pip install vllm>=0.14.0
    pip install -e git+https://github.com/fynnsu/vllm-hidden-states-extractor.git
    
    # Serve with KV connector for hidden states
    vllm serve <model_path> --kv-transfer-config '{
        "kv_connector":"ExampleHiddenStatesConnector",
        "kv_role":"kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": "/tmp/hidden_states"}
    }'
"""

import argparse
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
TOKENS_GENERATED = 256
MAX_PROMPT_LEN = 2048

# Sample prompts (Craftax observations)
SAMPLE_PROMPTS = [
    "Map (interesting tiles only): 0,1:tree, -1,0:stone\nInventory: Wood:0\nHealth:9\nDirection:right\n\nWhat action should you take?",
    "Map (interesting tiles only): 1,0:water, 0,-1:Cow on grass\nInventory: Wood:5\nHealth:7\nDirection:down\n\nWhat action should you take?",
    "Map (interesting tiles only): -1,-1:Skeleton on grass\nInventory: Wood:3, Stone:2\nHealth:5\nDirection:up\n\nWhat action should you take?",
    "Map: [No interesting tiles in view]\nInventory: Wood:10, Stone:5\nHealth:9\nDirection:left\n\nWhat action should you take?",
]


def get_test_prompts(n: int) -> List[str]:
    """Generate n test prompts by cycling through samples."""
    return [SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)] for i in range(n)]


# =============================================================================
# Method 1: HuggingFace + Flash Attention (Current Approach)
# =============================================================================

def benchmark_huggingface_flash(
    model_id: str,
    prompts: List[str],
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Benchmark HuggingFace with Flash Attention 2 and output_hidden_states.
    This is our current approach in online_rl_hidden.py.
    """
    print(f"\n{'='*60}")
    print("Method 1: HuggingFace + Flash Attention 2")
    print(f"{'='*60}")
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import gc
    
    # Load model
    print(f"Loading model: {model_id}")
    load_start = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        flash_enabled = True
    except Exception as e:
        print(f"Flash Attention not available: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        flash_enabled = False
    
    load_time = time.perf_counter() - load_start
    hidden_size = model.config.hidden_size
    print(f"Model loaded in {load_time:.1f}s (flash={flash_enabled}, hidden_size={hidden_size})")
    
    # Process batches
    all_hidden = []
    latencies = []
    total_start = time.perf_counter()
    
    for batch_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        batch_start = time.perf_counter()
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LEN,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=TOKENS_GENERATED,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=0.7,
            )
        
        # Extract and pool hidden states
        last_layer_states = [s[-1] for s in outputs.hidden_states]
        generated_hidden = torch.cat(last_layer_states, dim=1)
        
        seq_len = generated_hidden.shape[1]
        if seq_len > TOKENS_GENERATED:
            generated_hidden = generated_hidden[:, :TOKENS_GENERATED, :]
        elif seq_len < TOKENS_GENERATED:
            padding = torch.zeros(
                (len(batch_prompts), TOKENS_GENERATED - seq_len, hidden_size),
                device=generated_hidden.device,
                dtype=generated_hidden.dtype,
            )
            generated_hidden = torch.cat([generated_hidden, padding], dim=1)
        
        pooled = generated_hidden.mean(dim=1).cpu().numpy()
        all_hidden.append(pooled)
        
        batch_time = time.perf_counter() - batch_start
        latencies.append(batch_time)
        print(f"  Batch {batch_idx//batch_size + 1}: {batch_time:.2f}s ({len(batch_prompts)/batch_time:.2f} SPS)")
    
    total_time = time.perf_counter() - total_start
    all_hidden = np.concatenate(all_hidden, axis=0)
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "backend": "huggingface_flash",
        "flash_enabled": flash_enabled,
        "total_time_s": total_time,
        "samples_per_sec": len(prompts) / total_time,
        "mean_latency_s": np.mean(latencies),
        "hidden_shape": list(all_hidden.shape),
        "hidden_size": hidden_size,
        "tokens_generated": TOKENS_GENERATED,
        "num_samples": len(prompts),
    }


# =============================================================================
# Method 2: vLLM Hidden States Extractor Plugin
# =============================================================================

def benchmark_vllm_plugin(
    server_url: str,
    model_name: str,
    prompts: List[str],
    hidden_states_path: str = "/tmp/hidden_states",
) -> Dict[str, Any]:
    """
    Benchmark vLLM Hidden States Extractor Plugin.
    
    Requires:
    - vLLM 0.14.0+ with the plugin installed
    - Server running with KV connector configured
    
    The plugin saves hidden states to disk as safetensors files.
    We measure end-to-end time including file loading.
    """
    print(f"\n{'='*60}")
    print("Method 2: vLLM Hidden States Extractor Plugin")
    print(f"{'='*60}")
    
    try:
        import requests
        import safetensors.torch
    except ImportError as e:
        return {"backend": "vllm_plugin", "error": f"Missing dependency: {e}"}
    
    # Check server health
    print(f"Checking server at {server_url}...")
    try:
        health = requests.get(f"{server_url}/health", timeout=10)
        if health.status_code != 200:
            return {"backend": "vllm_plugin", "error": f"Server not healthy: {health.status_code}"}
    except Exception as e:
        return {"backend": "vllm_plugin", "error": f"Cannot connect to server: {e}. Start server with: vllm serve <model> --kv-transfer-config <config>"}
    
    print(f"Server healthy. Sending {len(prompts)} requests...")
    
    all_hidden = []
    latencies = []
    hidden_shapes = []
    
    total_start = time.perf_counter()
    
    for i, prompt in enumerate(prompts):
        req_start = time.perf_counter()
        
        # Send request to vLLM server
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": TOKENS_GENERATED,
            "temperature": 0.7,
        }
        
        try:
            response = requests.post(
                f"{server_url}/v1/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            print(f"  Request {i+1} failed: {e}")
            continue
        
        # Extract hidden states filepath from response
        kv_params = result.get("kv_transfer_params", {})
        hs_filepath = kv_params.get("hidden_states_path")
        
        if not hs_filepath:
            print(f"  Request {i+1}: No hidden_states_path in response")
            continue
        
        # Load hidden states from file
        try:
            data = safetensors.torch.load_file(hs_filepath)
            hidden_states = data["hidden_states"]  # Shape: [num_layers, seq_len, hidden_size]
            
            # Mean pool across layers and sequence
            # Take last layer, mean over sequence
            last_layer = hidden_states[-1]  # [seq_len, hidden_size]
            pooled = last_layer.mean(dim=0).numpy()  # [hidden_size]
            all_hidden.append(pooled)
            hidden_shapes.append(list(hidden_states.shape))
            
        except Exception as e:
            print(f"  Request {i+1}: Failed to load hidden states: {e}")
            continue
        
        latency = time.perf_counter() - req_start
        latencies.append(latency)
        
        if (i + 1) % 5 == 0:
            avg_lat = np.mean(latencies[-5:])
            print(f"  Requests {i+1}/{len(prompts)}: avg latency {avg_lat:.2f}s ({1/avg_lat:.2f} SPS)")
    
    total_time = time.perf_counter() - total_start
    
    if not all_hidden:
        return {"backend": "vllm_plugin", "error": "No successful requests"}
    
    all_hidden = np.stack(all_hidden, axis=0)
    
    return {
        "backend": "vllm_plugin",
        "total_time_s": total_time,
        "samples_per_sec": len(all_hidden) / total_time,
        "mean_latency_s": np.mean(latencies),
        "hidden_shape": list(all_hidden.shape),
        "raw_hidden_shapes": hidden_shapes[:3],  # Sample of shapes from server
        "num_samples": len(all_hidden),
        "num_failed": len(prompts) - len(all_hidden),
        "note": "Includes disk I/O for loading safetensors files",
    }


# =============================================================================
# Method 3: vLLM Plugin with Batched Requests
# =============================================================================

def benchmark_vllm_plugin_batched(
    server_url: str,
    model_name: str,
    prompts: List[str],
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Benchmark vLLM Plugin with concurrent/batched requests.
    Uses async requests to maximize throughput.
    """
    print(f"\n{'='*60}")
    print("Method 3: vLLM Plugin (Batched/Async)")
    print(f"{'='*60}")
    
    try:
        import requests
        import safetensors.torch
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except ImportError as e:
        return {"backend": "vllm_plugin_batched", "error": f"Missing dependency: {e}"}
    
    # Check server
    try:
        health = requests.get(f"{server_url}/health", timeout=10)
        if health.status_code != 200:
            return {"backend": "vllm_plugin_batched", "error": f"Server not healthy"}
    except Exception as e:
        return {"backend": "vllm_plugin_batched", "error": f"Cannot connect: {e}"}
    
    print(f"Sending {len(prompts)} requests with batch_size={batch_size}...")
    
    def send_request(prompt_idx_tuple):
        idx, prompt = prompt_idx_tuple
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": TOKENS_GENERATED,
            "temperature": 0.7,
        }
        
        try:
            response = requests.post(
                f"{server_url}/v1/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            
            hs_path = result.get("kv_transfer_params", {}).get("hidden_states_path")
            if hs_path:
                data = safetensors.torch.load_file(hs_path)
                hidden = data["hidden_states"][-1].mean(dim=0).numpy()
                return idx, hidden, None
            return idx, None, "No hidden_states_path"
        except Exception as e:
            return idx, None, str(e)
    
    results = [None] * len(prompts)
    errors = []
    
    total_start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(send_request, (i, p)): i 
                   for i, p in enumerate(prompts)}
        
        completed = 0
        for future in as_completed(futures):
            idx, hidden, error = future.result()
            if hidden is not None:
                results[idx] = hidden
            else:
                errors.append(error)
            
            completed += 1
            if completed % 5 == 0:
                elapsed = time.perf_counter() - total_start
                print(f"  Completed {completed}/{len(prompts)} ({completed/elapsed:.2f} SPS)")
    
    total_time = time.perf_counter() - total_start
    
    valid_hidden = [h for h in results if h is not None]
    if not valid_hidden:
        return {"backend": "vllm_plugin_batched", "error": "No successful requests"}
    
    all_hidden = np.stack(valid_hidden, axis=0)
    
    return {
        "backend": "vllm_plugin_batched",
        "total_time_s": total_time,
        "samples_per_sec": len(valid_hidden) / total_time,
        "hidden_shape": list(all_hidden.shape),
        "num_samples": len(valid_hidden),
        "num_failed": len(errors),
        "concurrent_requests": batch_size,
    }


# =============================================================================
# Results
# =============================================================================

def print_results(results: List[Dict[str, Any]]):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Backend':<25} {'Time (s)':<12} {'Samples/sec':<12} {'Status'}")
    print("-" * 70)
    
    for r in results:
        if "error" in r:
            print(f"{r['backend']:<25} {'N/A':<12} {'N/A':<12} ❌ {r['error'][:30]}...")
        else:
            time_s = r.get("total_time_s", 0)
            sps = r.get("samples_per_sec", 0)
            print(f"{r['backend']:<25} {time_s:<12.2f} {sps:<12.2f} ✅")
    
    print("-" * 70)
    
    # Find best
    valid = [r for r in results if "samples_per_sec" in r and r["samples_per_sec"] > 0]
    if valid:
        best = max(valid, key=lambda x: x["samples_per_sec"])
        print(f"\n🏆 WINNER: {best['backend']} at {best['samples_per_sec']:.2f} samples/sec")
        
        # Speedup comparison
        if len(valid) > 1:
            print("\nSpeedup factors:")
            baseline_sps = valid[0]["samples_per_sec"]
            for r in valid:
                speedup = r["samples_per_sec"] / baseline_sps
                print(f"  {r['backend']}: {speedup:.2f}x vs {valid[0]['backend']}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM hidden states plugin vs HuggingFace")
    parser.add_argument("--backend", type=str, default="all",
                        choices=["huggingface", "vllm-plugin", "vllm-plugin-batched", "all"])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm-model-name", type=str, default="./demo/qwen3_8b",
                        help="Model name as served by vLLM (may differ from HF model)")
    args = parser.parse_args()
    
    prompts = get_test_prompts(args.samples)
    print(f"Testing with {len(prompts)} samples")
    print(f"HuggingFace model: {args.model}")
    print(f"vLLM server: {args.vllm_url}")
    print(f"Tokens to generate: {TOKENS_GENERATED}")
    
    results = []
    
    if args.backend in ["huggingface", "all"]:
        result = benchmark_huggingface_flash(args.model, prompts, args.batch_size)
        results.append(result)
    
    if args.backend in ["vllm-plugin", "all"]:
        result = benchmark_vllm_plugin(
            args.vllm_url, 
            args.vllm_model_name, 
            prompts
        )
        results.append(result)
    
    if args.backend in ["vllm-plugin-batched", "all"]:
        result = benchmark_vllm_plugin_batched(
            args.vllm_url,
            args.vllm_model_name,
            prompts,
            args.batch_size,
        )
        results.append(result)
    
    print_results(results)
    
    # Save results
    output_file = f"bench_vllm_plugin_{args.samples}samples.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
