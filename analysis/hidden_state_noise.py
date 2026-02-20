#!/usr/bin/env python3
"""
Hidden State Noise Analysis

Tests whether LLM hidden state extraction produces meaningful signal or noise
by applying meaning-preserving transformations to observations and comparing
the resulting 2560-dim hidden state vectors.

Augmentations tested:
  1. 180° rotation: (r, c) → (-r, -c), direction flipped
  2. Horizontal reflection: (r, c) → (r, -c), left↔right
  3. Vertical reflection: (r, c) → (-r, c), up↔down
  4. Whitespace/formatting: extra spaces, number format tweaks

Additional signal-vs-noise tests:
  5. Repeated inference: same observation N times (noise floor)
  6. Temperature sweep: temp 0.0 vs 0.7 vs 1.0
  7. Different observations: cross-observation similarity (should be lower)
  8. Random text ablation: junk prompt vs real (should be very different)

Usage:
  python analysis/hidden_state_noise.py --num_obs 4 --num_repeats 3
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_prompts import filter_text_obs, create_prompt, SYSTEM_PROMPT, FEW_SHOT_EXAMPLES


# =============================================================================
# Augmentation Functions
# =============================================================================

def parse_map_line(text_obs: str) -> Tuple[str, List[Tuple[int, int, str]], str]:
    """Parse a filtered text observation into map tiles and non-map content.
    
    Returns:
        prefix: text before map line
        tiles: list of (row, col, tile_type)
        suffix: text after map line (inventory, stats, etc.)
    """
    lines = text_obs.split('\n')
    map_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Map'):
            map_line_idx = i
            break
    
    if map_line_idx is None:
        return text_obs, [], ""
    
    prefix = '\n'.join(lines[:map_line_idx])
    map_line = lines[map_line_idx].strip()
    suffix = '\n'.join(lines[map_line_idx + 1:])
    
    # Parse "Map (interesting tiles only): r1, c1:tile1, r2, c2:tile2, ..."
    # or "Map: ..." 
    colon_idx = map_line.index(':')
    map_header = map_line[:colon_idx + 1]
    tile_str = map_line[colon_idx + 1:].strip()
    
    tiles = []
    if tile_str and not tile_str.startswith('['):
        # Parse tiles: "-3, -4:tree, -1, -4:tree, ..."
        # Each tile is "r, c:type" — but commas also separate tiles
        # Strategy: split by the pattern that identifies tile boundaries
        # A tile boundary is ", " followed by a coordinate pattern like "-?\d+"
        # But ", " also appears within a tile coord like "-3, -4:tree"
        # Better: use regex to find all "r, c:type" patterns
        tile_pattern = re.findall(r'(-?\d+),\s*(-?\d+):([^,]+?)(?=,\s*-?\d+,\s*-?\d+:|$)', tile_str)
        for r, c, tile_type in tile_pattern:
            tiles.append((int(r), int(c), tile_type.strip()))
    
    return (prefix, map_header, tiles, suffix)


def reconstruct_obs(prefix: str, map_header: str, tiles: List[Tuple[int, int, str]], suffix: str) -> str:
    """Reconstruct a text observation from parsed components."""
    if not tiles:
        map_line = f"{map_header} [No interesting tiles in view - all background]"
    else:
        tile_strs = [f"{r}, {c}:{t}" for r, c, t in tiles]
        map_line = f"{map_header} {', '.join(tile_strs)}"
    
    parts = []
    if prefix.strip():
        parts.append(prefix)
    parts.append(map_line)
    if suffix.strip():
        parts.append(suffix)
    return '\n'.join(parts)


def flip_direction(direction: str, augmentation: str) -> str:
    """Transform the Direction field for a given augmentation."""
    if augmentation == "rotate_180":
        mapping = {"left": "right", "right": "left", "up": "down", "down": "up"}
    elif augmentation == "reflect_horizontal":
        mapping = {"left": "right", "right": "left", "up": "up", "down": "down"}
    elif augmentation == "reflect_vertical":
        mapping = {"left": "left", "right": "right", "up": "down", "down": "up"}
    else:
        return direction
    return mapping.get(direction.lower(), direction)


def transform_direction_in_text(text: str, augmentation: str) -> str:
    """Find and transform the Direction: line in the observation text."""
    lines = text.split('\n')
    result = []
    for line in lines:
        if line.strip().startswith('Direction:'):
            direction = line.split(':', 1)[1].strip()
            new_direction = flip_direction(direction, augmentation)
            result.append(f"Direction: {new_direction}")
        else:
            result.append(line)
    return '\n'.join(result)


def augment_rotate_180(text_obs: str) -> str:
    """180° rotation: (r, c) → (-r, -c), flip direction."""
    parsed = parse_map_line(text_obs)
    if isinstance(parsed, str):
        return parsed
    prefix, map_header, tiles, suffix = parsed
    
    rotated_tiles = [(-r, -c, t) for r, c, t in tiles]
    # Sort by (r, c) for consistency
    rotated_tiles.sort(key=lambda x: (x[0], x[1]))
    
    result = reconstruct_obs(prefix, map_header, rotated_tiles, suffix)
    return transform_direction_in_text(result, "rotate_180")


def augment_reflect_horizontal(text_obs: str) -> str:
    """Horizontal reflection: (r, c) → (r, -c), flip left↔right."""
    parsed = parse_map_line(text_obs)
    if isinstance(parsed, str):
        return parsed
    prefix, map_header, tiles, suffix = parsed
    
    reflected_tiles = [(r, -c, t) for r, c, t in tiles]
    reflected_tiles.sort(key=lambda x: (x[0], x[1]))
    
    result = reconstruct_obs(prefix, map_header, reflected_tiles, suffix)
    return transform_direction_in_text(result, "reflect_horizontal")


def augment_reflect_vertical(text_obs: str) -> str:
    """Vertical reflection: (r, c) → (-r, c), flip up↔down."""
    parsed = parse_map_line(text_obs)
    if isinstance(parsed, str):
        return parsed
    prefix, map_header, tiles, suffix = parsed
    
    reflected_tiles = [(-r, c, t) for r, c, t in tiles]
    reflected_tiles.sort(key=lambda x: (x[0], x[1]))
    
    result = reconstruct_obs(prefix, map_header, reflected_tiles, suffix)
    return transform_direction_in_text(result, "reflect_vertical")


def augment_whitespace(text_obs: str) -> str:
    """Whitespace/formatting changes that don't alter meaning."""
    result = text_obs
    # Add extra space after colons in inventory
    result = result.replace("Wood: ", "Wood:  ")
    result = result.replace("Stone: ", "Stone:  ")
    # Change "9.0" → "9" for health-like values
    result = re.sub(r'Health: (\d+)\.0\b', r'Health: \1', result)
    # Add trailing spaces to some lines
    lines = result.split('\n')
    modified = []
    for i, line in enumerate(lines):
        if i % 3 == 0 and line.strip():
            modified.append(line + "  ")
        else:
            modified.append(line)
    return '\n'.join(modified)


AUGMENTATIONS = {
    "rotate_180": augment_rotate_180,
    "reflect_horizontal": augment_reflect_horizontal,
    "reflect_vertical": augment_reflect_vertical,
    "whitespace": augment_whitespace,
}


# =============================================================================
# Observation Loading
# =============================================================================

def load_observations_from_golden(num_obs: int = 4) -> List[str]:
    """Load filtered text observations from golden_examples/ JSONL files."""
    golden_dir = PROJECT_ROOT / "golden_examples"
    jsonl_files = sorted(golden_dir.glob("*/examples.jsonl"))
    
    if not jsonl_files:
        raise FileNotFoundError(f"No examples.jsonl files found in {golden_dir}")
    
    observations = []
    for jsonl_path in jsonl_files:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                # Use the filtered observation — same as what the pipeline sees
                obs = example.get("before_state_filtered", "")
                if obs:
                    observations.append(obs)
                if len(observations) >= num_obs:
                    break
        if len(observations) >= num_obs:
            break
    
    print(f"Loaded {len(observations)} observations from golden_examples/")
    return observations[:num_obs]


# =============================================================================
# Hidden State Extraction
# =============================================================================

class HiddenStateExtractor:
    """Thin wrapper around LLM for extracting hidden states.
    
    Uses the same prompt formatting as the production pipeline
    (create_prompt from utils/llm_prompts.py).
    """
    
    def __init__(self, model_id: str = "Qwen/Qwen3-4B-Thinking-2507", 
                 tokens_generated: int = 256):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokens_generated = tokens_generated
        self.model_id = model_id
        
        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            print("Using Flash Attention 2")
        except Exception as e:
            print(f"Flash Attention not available ({e}), using default")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        self.hidden_size = self.model.config.hidden_size
        print(f"Model loaded. Hidden size: {self.hidden_size}")
    
    def extract(self, observations: List[str], temperature: float = 0.7,
                batch_size: int = 4) -> Tuple[np.ndarray, List[str]]:
        """Extract hidden states from text observations.
        
        Uses create_prompt() for prompt formatting — identical to production.
        
        Returns:
            hidden_states: (N, hidden_size) mean-pooled hidden states
            generated_texts: list of generated text strings
        """
        all_hidden = []
        all_texts = []
        
        for batch_start in range(0, len(observations), batch_size):
            batch_obs = observations[batch_start:batch_start + batch_size]
            
            # Format prompts exactly like production pipeline
            prompts = [create_prompt(obs, self.tokenizer) for obs in batch_obs]
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.model.device)
            
            prompt_len = inputs['input_ids'].shape[1]
            
            gen_kwargs = dict(
                max_new_tokens=self.tokens_generated,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            
            if temperature == 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Extract last-layer hidden states per generation step
            # Same logic as llm_extractor.py lines 199-200
            hs_per_step = [step_hs[-1][:, -1, :] for step_hs in outputs.hidden_states]
            generated_hidden = torch.stack(hs_per_step, dim=1)  # (batch, seq, hidden)
            
            # Pad/truncate to tokens_generated
            seq_len = generated_hidden.shape[1]
            if seq_len > self.tokens_generated:
                generated_hidden = generated_hidden[:, :self.tokens_generated, :]
            elif seq_len < self.tokens_generated:
                pad = torch.zeros(
                    (len(batch_obs), self.tokens_generated - seq_len, self.hidden_size),
                    device=generated_hidden.device, dtype=generated_hidden.dtype
                )
                generated_hidden = torch.cat([generated_hidden, pad], dim=1)
            
            # Mean pool → (batch, hidden_size)
            pooled = generated_hidden.mean(dim=1).cpu().numpy().astype(np.float32)
            all_hidden.append(pooled)
            
            # Decode text
            gen_ids = outputs.sequences[:, prompt_len:]
            texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_texts.extend(texts)
        
        return np.concatenate(all_hidden, axis=0), all_texts


# =============================================================================
# Analysis Utilities
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def print_comparison_table(results: Dict[str, List[float]], metric_name: str):
    """Pretty-print a table of comparison results."""
    print(f"\n{'=' * 70}")
    print(f"  {metric_name}")
    print(f"{'=' * 70}")
    print(f"  {'Comparison':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-' * 67}")
    for name, values in results.items():
        arr = np.array(values)
        print(f"  {name:<35} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f}")
    print()


# =============================================================================
# Main Analysis
# =============================================================================

def run_analysis(args):
    print("=" * 70)
    print("  Hidden State Noise Analysis")
    print("=" * 70)
    
    # 1. Load observations
    print("\n[1/6] Loading observations...")
    observations = load_observations_from_golden(args.num_obs)
    for i, obs in enumerate(observations):
        preview = obs[:120].replace('\n', ' | ')
        print(f"  Obs {i}: {preview}...")
    
    # 2. Initialize extractor
    print(f"\n[2/6] Loading model ({args.model_id})...")
    extractor = HiddenStateExtractor(
        model_id=args.model_id,
        tokens_generated=args.tokens_generated,
    )
    
    # 3. Generate augmented observations
    print("\n[3/6] Generating augmented observations...")
    augmented_obs = {}
    for aug_name, aug_fn in AUGMENTATIONS.items():
        augmented_obs[aug_name] = [aug_fn(obs) for obs in observations]
        # Show a sample
        print(f"\n  --- {aug_name} (observation 0) ---")
        orig_preview = observations[0][:200].replace('\n', ' | ')
        aug_preview = augmented_obs[aug_name][0][:200].replace('\n', ' | ')
        print(f"  Original:  {orig_preview}")
        print(f"  Augmented: {aug_preview}")
    
    # 4. Extract hidden states
    print("\n[4/6] Extracting hidden states...")
    
    # --- Original observations ---
    print("  Extracting: original observations...")
    t0 = time.time()
    hs_original, texts_original = extractor.extract(observations, temperature=0.7)
    print(f"    Done in {time.time() - t0:.1f}s. Shape: {hs_original.shape}")
    
    # --- Repeated inference (same observation, N times) ---
    print(f"  Extracting: repeated inference ({args.num_repeats} repeats)...")
    hs_repeats = []
    for rep in range(args.num_repeats):
        t0 = time.time()
        hs_rep, _ = extractor.extract(observations, temperature=0.7)
        hs_repeats.append(hs_rep)
        print(f"    Repeat {rep+1}/{args.num_repeats} done in {time.time() - t0:.1f}s")
    
    # --- Augmented observations ---
    hs_augmented = {}
    for aug_name, aug_list in augmented_obs.items():
        print(f"  Extracting: {aug_name}...")
        t0 = time.time()
        hs_aug, _ = extractor.extract(aug_list, temperature=0.7)
        hs_augmented[aug_name] = hs_aug
        print(f"    Done in {time.time() - t0:.1f}s")
    
    # --- Temperature sweep ---
    print("  Extracting: temperature=0.0 (greedy)...")
    t0 = time.time()
    hs_temp0, _ = extractor.extract(observations, temperature=0.0)
    print(f"    Done in {time.time() - t0:.1f}s")
    
    print("  Extracting: temperature=1.0...")
    t0 = time.time()
    hs_temp1, _ = extractor.extract(observations, temperature=1.0)
    print(f"    Done in {time.time() - t0:.1f}s")
    
    # --- Random text ablation ---
    print("  Extracting: random/junk text...")
    junk_observations = [
        "The quick brown fox jumps over the lazy dog. This is completely unrelated text.",
        "Hello world. Random numbers: 42, 17, 99. This is not a game observation.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. No game data here.",
        "Weather forecast: sunny with a chance of rain. Temperature 72F.",
    ][:len(observations)]
    t0 = time.time()
    hs_junk, _ = extractor.extract(junk_observations, temperature=0.7)
    print(f"    Done in {time.time() - t0:.1f}s")
    
    # 5. Compute metrics
    print("\n[5/6] Computing metrics...")
    
    cos_results = {}
    l2_results = {}
    
    # Repeated inference: original vs repeats (noise floor)
    repeat_cos = []
    repeat_l2 = []
    for hs_rep in hs_repeats:
        for i in range(len(observations)):
            repeat_cos.append(cosine_similarity(hs_original[i], hs_rep[i]))
            repeat_l2.append(l2_distance(hs_original[i], hs_rep[i]))
    cos_results["Repeated inference (noise floor)"] = repeat_cos
    l2_results["Repeated inference (noise floor)"] = repeat_l2
    
    # Augmented vs original (same observation)
    for aug_name, hs_aug in hs_augmented.items():
        aug_cos = []
        aug_l2 = []
        for i in range(len(observations)):
            aug_cos.append(cosine_similarity(hs_original[i], hs_aug[i]))
            aug_l2.append(l2_distance(hs_original[i], hs_aug[i]))
        cos_results[f"Augmented: {aug_name}"] = aug_cos
        l2_results[f"Augmented: {aug_name}"] = aug_l2
    
    # Temperature: original (0.7) vs temp=0
    temp0_cos = []
    temp0_l2 = []
    for i in range(len(observations)):
        temp0_cos.append(cosine_similarity(hs_original[i], hs_temp0[i]))
        temp0_l2.append(l2_distance(hs_original[i], hs_temp0[i]))
    cos_results["Temperature: 0.7 vs 0.0"] = temp0_cos
    l2_results["Temperature: 0.7 vs 0.0"] = temp0_l2
    
    # Temperature: original (0.7) vs temp=1.0
    temp1_cos = []
    temp1_l2 = []
    for i in range(len(observations)):
        temp1_cos.append(cosine_similarity(hs_original[i], hs_temp1[i]))
        temp1_l2.append(l2_distance(hs_original[i], hs_temp1[i]))
    cos_results["Temperature: 0.7 vs 1.0"] = temp1_cos
    l2_results["Temperature: 0.7 vs 1.0"] = temp1_l2
    
    # Cross-observation: different observations (should be lower similarity)
    cross_cos = []
    cross_l2 = []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            cross_cos.append(cosine_similarity(hs_original[i], hs_original[j]))
            cross_l2.append(l2_distance(hs_original[i], hs_original[j]))
    cos_results["Different observations"] = cross_cos
    l2_results["Different observations"] = cross_l2
    
    # Random text vs real
    junk_cos = []
    junk_l2 = []
    for i in range(len(observations)):
        junk_cos.append(cosine_similarity(hs_original[i], hs_junk[i]))
        junk_l2.append(l2_distance(hs_original[i], hs_junk[i]))
    cos_results["Random text vs real"] = junk_cos
    l2_results["Random text vs real"] = junk_l2
    
    # 6. Print results
    print("\n[6/6] Results")
    print_comparison_table(cos_results, "Cosine Similarity (higher = more similar)")
    print_comparison_table(l2_results, "L2 Distance (lower = more similar)")
    
    # Interpretation guide
    print("=" * 70)
    print("  INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
  If hidden states carry MEANINGFUL SIGNAL:
    ✓ Repeated inference cos-sim should be high (>0.9)
    ✓ Meaning-preserving augmentations should be close to repeated inference
    ✓ Different observations should have LOWER cos-sim than augmented same-obs
    ✓ Random text should have LOWEST cos-sim
    ✓ Temperature 0.0 vs 0.7 should show that sampling adds noise

  If hidden states are mostly NOISE:
    ✗ All comparisons will have similar cos-sim (no discrimination)
    ✗ Augmented same-obs ≈ different obs ≈ random text
    ✗ Repeated inference will have low cos-sim (high variance)
""")
    
    # Save results
    results_dir = PROJECT_ROOT / "analysis" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"noise_analysis_{timestamp}.npz"
    
    save_data = {
        "hs_original": hs_original,
        "hs_temp0": hs_temp0,
        "hs_temp1": hs_temp1,
        "hs_junk": hs_junk,
    }
    for aug_name, hs_aug in hs_augmented.items():
        save_data[f"hs_{aug_name}"] = hs_aug
    for i, hs_rep in enumerate(hs_repeats):
        save_data[f"hs_repeat_{i}"] = hs_rep
    
    np.savez(str(results_path), **save_data)
    print(f"  Raw hidden states saved to: {results_path}")
    
    # Also save metrics as JSON
    metrics_path = results_dir / f"noise_analysis_{timestamp}.json"
    metrics = {
        "cosine_similarity": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), 
                                   "values": [float(x) for x in v]} 
                               for k, v in cos_results.items()},
        "l2_distance": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                            "values": [float(x) for x in v]}
                        for k, v in l2_results.items()},
        "config": {
            "model_id": args.model_id,
            "tokens_generated": args.tokens_generated,
            "num_obs": args.num_obs,
            "num_repeats": args.num_repeats,
            "hidden_size": extractor.hidden_size,
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidden State Noise Analysis")
    parser.add_argument("--num_obs", type=int, default=4,
                        help="Number of observations to test")
    parser.add_argument("--num_repeats", type=int, default=3,
                        help="Number of repeated inferences per observation")
    parser.add_argument("--tokens_generated", type=int, default=256,
                        help="Tokens to generate (matches production)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Model to use")
    args = parser.parse_args()
    
    run_analysis(args)
