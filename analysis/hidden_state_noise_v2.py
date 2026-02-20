#!/usr/bin/env python3
"""
Hidden State Noise Analysis v2

Follow-up experiments to determine if LLM hidden states carry meaningful signal.
Based on v1 findings that all comparisons cluster at ~0.95 cosine similarity.

New experiments:
  1. Mean subtraction — subtract the "background" hidden state vector
  2. Prompt-only forward pass — no generation, just encode the prompt
  3. PCA on residuals — visualize structure after mean removal
  4. Greedy decoding only — eliminate sampling noise entirely
  5. Re-run all augmentation comparisons on the improved representations

Usage:
  python analysis/hidden_state_noise_v2.py --num_obs 4 --num_repeats 3
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_prompts import (
    filter_text_obs, create_prompt, create_chat_messages,
    SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
)

# Import augmentations from v1
from analysis.hidden_state_noise import (
    AUGMENTATIONS, load_observations_from_golden,
    cosine_similarity, l2_distance, print_comparison_table,
)


# =============================================================================
# Enhanced Extractor with Prompt-Only and Mean Subtraction
# =============================================================================

class EnhancedExtractor:
    """Extended extractor supporting prompt-only hidden states and mean subtraction."""

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

    def extract_generative(self, observations: List[str], temperature: float = 0.7,
                           batch_size: int = 1) -> Tuple[np.ndarray, List[str]]:
        """Extract hidden states via generation (same as production pipeline).
        
        Uses batch_size=1 by default to avoid flash attention batching artifacts
        that can cause identical hidden states across batch elements.

        Returns:
            hidden_states: (N, hidden_size) mean-pooled hidden states
            generated_texts: list of generated text strings
        """
        all_hidden = []
        all_texts = []

        for batch_start in range(0, len(observations), batch_size):
            batch_obs = observations[batch_start:batch_start + batch_size]
            prompts = [create_prompt(obs, self.tokenizer) for obs in batch_obs]

            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=False,
            ).to(self.model.device)
            
            if batch_start == 0:  # Debug: print token count for first batch
                print(f"    Prompt token count: {inputs['input_ids'].shape[1]}")

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

            # Same extraction as llm_extractor.py
            hs_per_step = [step_hs[-1][:, -1, :] for step_hs in outputs.hidden_states]
            generated_hidden = torch.stack(hs_per_step, dim=1)

            seq_len = generated_hidden.shape[1]
            if seq_len > self.tokens_generated:
                generated_hidden = generated_hidden[:, :self.tokens_generated, :]
            elif seq_len < self.tokens_generated:
                pad = torch.zeros(
                    (len(batch_obs), self.tokens_generated - seq_len, self.hidden_size),
                    device=generated_hidden.device, dtype=generated_hidden.dtype
                )
                generated_hidden = torch.cat([generated_hidden, pad], dim=1)

            pooled = generated_hidden.mean(dim=1).cpu().numpy().astype(np.float32)
            all_hidden.append(pooled)

            gen_ids = outputs.sequences[:, prompt_len:]
            texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_texts.extend(texts)

        return np.concatenate(all_hidden, axis=0), all_texts

    def extract_last_generated_token(self, observations: List[str], temperature: float = 0.7,
                                      batch_size: int = 1) -> Tuple[np.ndarray, List[str]]:
        """Extract hidden state from the LAST generated token only.
        
        Generates text (with CoT reasoning), but instead of mean-pooling all
        256 hidden states, takes only the final token's hidden state. This
        captures the model's state after completing its reasoning chain.

        Returns:
            hidden_states: (N, hidden_size) last-token hidden states
            generated_texts: list of generated text strings
        """
        all_hidden = []
        all_texts = []

        for batch_start in range(0, len(observations), batch_size):
            batch_obs = observations[batch_start:batch_start + batch_size]
            prompts = [create_prompt(obs, self.tokenizer) for obs in batch_obs]

            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=False,
            ).to(self.model.device)
            
            if batch_start == 0:
                print(f"    Prompt token count: {inputs['input_ids'].shape[1]}")

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

            # Extract hidden state from the LAST generation step only
            # outputs.hidden_states[-1] is the last step's hidden states tuple
            # [-1] gets the last layer, [:, -1, :] gets the last token position
            last_step_hs = outputs.hidden_states[-1][-1][:, -1, :]  # (batch, hidden)
            last_token_hidden = last_step_hs.cpu().numpy().astype(np.float32)
            all_hidden.append(last_token_hidden)

            gen_ids = outputs.sequences[:, prompt_len:]
            texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_texts.extend(texts)

        return np.concatenate(all_hidden, axis=0), all_texts

    def extract_prompt_only(self, observations: List[str],
                            batch_size: int = 1,
                            layer: int = -1) -> np.ndarray:
        """Extract hidden states from the prompt ONLY (no generation).

        Does a single forward pass and takes the last token's hidden state
        from the specified layer. This is deterministic — no sampling noise.
        
        Uses batch_size=1 by default to avoid left-padding position bugs
        and flash attention batching artifacts.

        Args:
            observations: List of filtered text observations
            batch_size: Batch size for processing
            layer: Which layer to extract from. -1 = last layer.
                   outputs.hidden_states has (num_layers+1) entries:
                   [0] = embedding, [1..36] = transformer layers 1-36.

        Returns:
            hidden_states: (N, hidden_size) 
        """
        all_hidden = []

        for batch_start in range(0, len(observations), batch_size):
            batch_obs = observations[batch_start:batch_start + batch_size]
            prompts = [create_prompt(obs, self.tokenizer) for obs in batch_obs]

            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=False,
            ).to(self.model.device)
            
            if batch_start == 0:  # Debug: print token count for first batch
                print(f"    Prompt token count: {inputs['input_ids'].shape[1]}")

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # outputs.hidden_states is tuple of (num_layers+1) tensors
            # [0] = embedding, [1..36] = transformer layers 1-36
            selected_layer = outputs.hidden_states[layer]  # (batch, seq_len, hidden)

            # With left padding, the last real token is ALWAYS at position -1
            pooled = selected_layer[:, -1, :].cpu().numpy().astype(np.float32)
            all_hidden.append(pooled)

        return np.concatenate(all_hidden, axis=0)

    def extract_background(self) -> np.ndarray:
        """Extract the 'background' hidden state from just the system prompt + few-shot.
        
        This is what the model's hidden state looks like before seeing ANY observation.
        Subtracting this should isolate observation-specific signal.

        Returns:
            background: (hidden_size,) vector
        """
        # Create a prompt with a dummy/empty observation
        dummy_obs = "Map: [No interesting tiles in view - all background]\nInventory:\nHealth: 9.0\nFood: 9\nDrink: 9\nEnergy: 9\nDirection: right"
        hs = self.extract_prompt_only([dummy_obs], batch_size=1)
        return hs[0]


# =============================================================================
# Analysis Utilities
# =============================================================================

def mean_subtract(hidden_states: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Subtract the background vector from all hidden states."""
    return hidden_states - background[np.newaxis, :]


def compute_all_metrics(hs_a: np.ndarray, hs_b: np.ndarray) -> Dict[str, List[float]]:
    """Compute cosine similarity and L2 between paired vectors."""
    cos_vals = []
    l2_vals = []
    for i in range(len(hs_a)):
        cos_vals.append(cosine_similarity(hs_a[i], hs_b[i]))
        l2_vals.append(l2_distance(hs_a[i], hs_b[i]))
    return {"cos": cos_vals, "l2": l2_vals}


def compute_cross_metrics(hs: np.ndarray) -> Dict[str, List[float]]:
    """Compute cross-observation (all-pairs) metrics."""
    cos_vals = []
    l2_vals = []
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            cos_vals.append(cosine_similarity(hs[i], hs[j]))
            l2_vals.append(l2_distance(hs[i], hs[j]))
    return {"cos": cos_vals, "l2": l2_vals}


def run_comparison_suite(hs_original, hs_repeats, hs_augmented, hs_junk,
                         label: str) -> Tuple[Dict, Dict]:
    """Run full comparison suite and print results. Returns cos/l2 result dicts."""
    cos_results = {}
    l2_results = {}

    # Repeated inference
    repeat_cos, repeat_l2 = [], []
    for hs_rep in hs_repeats:
        m = compute_all_metrics(hs_original, hs_rep)
        repeat_cos.extend(m["cos"])
        repeat_l2.extend(m["l2"])
    cos_results["Repeated inference (noise floor)"] = repeat_cos
    l2_results["Repeated inference (noise floor)"] = repeat_l2

    # Augmentations
    for aug_name, hs_aug in hs_augmented.items():
        m = compute_all_metrics(hs_original, hs_aug)
        cos_results[f"Augmented: {aug_name}"] = m["cos"]
        l2_results[f"Augmented: {aug_name}"] = m["l2"]

    # Cross-observation
    m = compute_cross_metrics(hs_original)
    cos_results["Different observations"] = m["cos"]
    l2_results["Different observations"] = m["l2"]

    # Random text
    m = compute_all_metrics(hs_original, hs_junk)
    cos_results["Random text vs real"] = m["cos"]
    l2_results["Random text vs real"] = m["l2"]

    print_comparison_table(cos_results, f"Cosine Similarity — {label}")
    print_comparison_table(l2_results, f"L2 Distance — {label}")

    return cos_results, l2_results


def save_pca_plot(hidden_states_dict: Dict[str, np.ndarray], save_path: str,
                  title: str = "PCA of Hidden States"):
    """Save a PCA scatter plot of hidden states from different conditions."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print(f"  [SKIP] matplotlib/sklearn not available, skipping PCA plot")
        return

    # Collect all vectors with labels
    all_vectors = []
    all_labels = []
    for label, hs in hidden_states_dict.items():
        for i in range(len(hs)):
            all_vectors.append(hs[i])
            all_labels.append(f"{label}[{i}]")

    all_vectors = np.array(all_vectors)

    if len(all_vectors) < 3:
        print("  [SKIP] Too few vectors for PCA")
        return

    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_vectors)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Color by condition
    unique_conditions = list(hidden_states_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_conditions)))

    idx = 0
    for ci, (label, hs) in enumerate(hidden_states_dict.items()):
        n = len(hs)
        ax.scatter(projected[idx:idx+n, 0], projected[idx:idx+n, 1],
                   c=[colors[ci]], label=label, s=80, alpha=0.7, edgecolors='black')
        # Annotate with index
        for j in range(n):
            ax.annotate(str(j), (projected[idx+j, 0], projected[idx+j, 1]),
                        fontsize=8, ha='center', va='bottom')
        idx += n

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PCA plot saved to: {save_path}")


# =============================================================================
# Main Analysis
# =============================================================================

def run_analysis(args):
    print("=" * 70)
    print("  Hidden State Noise Analysis v2")
    print("  (Mean Subtraction + Prompt-Only + PCA)")
    print("=" * 70)

    # 1. Load observations
    print("\n[1/8] Loading observations...")
    observations = load_observations_from_golden(args.num_obs)
    for i, obs in enumerate(observations):
        preview = obs[:100].replace('\n', ' | ')
        print(f"  Obs {i}: {preview}...")

    # 2. Initialize extractor
    print(f"\n[2/8] Loading model ({args.model_id})...")
    extractor = EnhancedExtractor(
        model_id=args.model_id,
        tokens_generated=args.tokens_generated,
    )

    # 3. Generate augmented observations
    print("\n[3/8] Generating augmented observations...")
    augmented_obs = {}
    for aug_name, aug_fn in AUGMENTATIONS.items():
        augmented_obs[aug_name] = [aug_fn(obs) for obs in observations]

    junk_observations = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world. Random numbers: 42, 17, 99.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Weather forecast: sunny with a chance of rain.",
    ][:len(observations)]

    # 4. Extract background hidden state
    print("\n[4/8] Extracting background hidden state (system prompt only)...")
    background = extractor.extract_background()
    bg_norm = np.linalg.norm(background)
    print(f"  Background vector norm: {bg_norm:.2f}")

    # =========================================================================
    # EXPERIMENT A: Generative with greedy decoding (temp=0) 
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT A: Generative, Greedy (temp=0)")
    print("=" * 70)

    print("\n[5/8] Extracting hidden states (greedy)...")
    
    print("  Original observations...")
    t0 = time.time()
    hs_orig_greedy, texts_orig = extractor.extract_generative(observations, temperature=0.0)
    print(f"    Done in {time.time() - t0:.1f}s")
    
    # Debug: verify hidden states actually differ
    print("\n  [DEBUG] Hidden state sanity check (greedy originals):")
    for i in range(len(observations)):
        norm = np.linalg.norm(hs_orig_greedy[i])
        vals = hs_orig_greedy[i][:5]
        text_preview = texts_orig[i][:80].replace('\n', ' ')
        print(f"    Obs {i}: norm={norm:.4f}, first5={vals}, text='{text_preview}'")
    if len(observations) >= 2:
        cs = cosine_similarity(hs_orig_greedy[0], hs_orig_greedy[1])
        l2 = l2_distance(hs_orig_greedy[0], hs_orig_greedy[1])
        print(f"    Obs0 vs Obs1: cos={cs:.6f}, l2={l2:.6f}")

    # Repeats (with greedy, should be deterministic)
    print(f"\n  Repeated inference ({args.num_repeats} repeats, greedy)...")
    hs_repeats_greedy = []
    for rep in range(args.num_repeats):
        t0 = time.time()
        hs_rep, _ = extractor.extract_generative(observations, temperature=0.0)
        hs_repeats_greedy.append(hs_rep)
        print(f"    Repeat {rep+1}/{args.num_repeats} done in {time.time() - t0:.1f}s")

    # Augmented
    hs_aug_greedy = {}
    for aug_name, aug_list in augmented_obs.items():
        print(f"  {aug_name}...")
        t0 = time.time()
        hs_aug, _ = extractor.extract_generative(aug_list, temperature=0.0)
        hs_aug_greedy[aug_name] = hs_aug
        print(f"    Done in {time.time() - t0:.1f}s")

    # Junk
    print("  Junk text...")
    hs_junk_greedy, texts_junk = extractor.extract_generative(junk_observations, temperature=0.0)
    
    # Debug: verify junk is different
    print("\n  [DEBUG] Junk text hidden states:")
    for i in range(len(junk_observations)):
        norm = np.linalg.norm(hs_junk_greedy[i])
        vals = hs_junk_greedy[i][:5]
        text_preview = texts_junk[i][:80].replace('\n', ' ')
        print(f"    Junk {i}: norm={norm:.4f}, first5={vals}, text='{text_preview}'")
    if len(observations) >= 1:
        cs = cosine_similarity(hs_orig_greedy[0], hs_junk_greedy[0])
        print(f"    Orig0 vs Junk0: cos={cs:.6f}")

    # Results: raw
    print("\n--- A.1: Raw hidden states (greedy) ---")
    cos_a1, l2_a1 = run_comparison_suite(
        hs_orig_greedy, hs_repeats_greedy, hs_aug_greedy, hs_junk_greedy,
        "Generative Greedy (raw)"
    )

    # Results: mean-subtracted
    print("\n--- A.2: Mean-subtracted hidden states (greedy) ---")
    hs_orig_greedy_ms = mean_subtract(hs_orig_greedy, background)
    hs_repeats_greedy_ms = [mean_subtract(h, background) for h in hs_repeats_greedy]
    hs_aug_greedy_ms = {k: mean_subtract(v, background) for k, v in hs_aug_greedy.items()}
    hs_junk_greedy_ms = mean_subtract(hs_junk_greedy, background)

    cos_a2, l2_a2 = run_comparison_suite(
        hs_orig_greedy_ms, hs_repeats_greedy_ms, hs_aug_greedy_ms, hs_junk_greedy_ms,
        "Generative Greedy (mean-subtracted)"
    )

    # =========================================================================
    # EXPERIMENT B: Prompt-only forward pass (deterministic, no generation)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: Prompt-Only Forward Pass (no generation)")
    print("=" * 70)

    print("\n[6/8] Extracting prompt-only hidden states...")

    print("  Original observations...")
    t0 = time.time()
    hs_orig_prompt = extractor.extract_prompt_only(observations)
    print(f"    Done in {time.time() - t0:.1f}s")
    
    # Debug: verify prompt-only hidden states differ
    print("\n  [DEBUG] Prompt-only sanity check:")
    for i in range(len(observations)):
        norm = np.linalg.norm(hs_orig_prompt[i])
        vals = hs_orig_prompt[i][:5]
        print(f"    Obs {i}: norm={norm:.4f}, first5={vals}")
    if len(observations) >= 2:
        cs = cosine_similarity(hs_orig_prompt[0], hs_orig_prompt[1])
        l2 = l2_distance(hs_orig_prompt[0], hs_orig_prompt[1])
        print(f"    Obs0 vs Obs1: cos={cs:.6f}, l2={l2:.6f}")

    # Repeats (should be perfectly deterministic)
    print(f"\n  Repeated inference ({args.num_repeats} repeats)...")
    hs_repeats_prompt = []
    for rep in range(args.num_repeats):
        t0 = time.time()
        hs_rep = extractor.extract_prompt_only(observations)
        hs_repeats_prompt.append(hs_rep)
        print(f"    Repeat {rep+1}/{args.num_repeats} done in {time.time() - t0:.1f}s")

    # Augmented
    hs_aug_prompt = {}
    for aug_name, aug_list in augmented_obs.items():
        print(f"  {aug_name}...")
        t0 = time.time()
        hs_aug = extractor.extract_prompt_only(aug_list)
        hs_aug_prompt[aug_name] = hs_aug
        print(f"    Done in {time.time() - t0:.1f}s")

    # Junk
    print("  Junk text...")
    hs_junk_prompt = extractor.extract_prompt_only(junk_observations)
    
    # Debug: verify junk prompt-only differs
    print("\n  [DEBUG] Prompt-only junk sanity check:")
    for i in range(len(junk_observations)):
        norm = np.linalg.norm(hs_junk_prompt[i])
        vals = hs_junk_prompt[i][:5]
        print(f"    Junk {i}: norm={norm:.4f}, first5={vals}")
    if len(observations) >= 1:
        cs = cosine_similarity(hs_orig_prompt[0], hs_junk_prompt[0])
        print(f"    Orig0 vs Junk0: cos={cs:.6f}")

    # Results: raw
    print("\n--- B.1: Raw hidden states (prompt-only) ---")
    cos_b1, l2_b1 = run_comparison_suite(
        hs_orig_prompt, hs_repeats_prompt, hs_aug_prompt, hs_junk_prompt,
        "Prompt-Only (raw)"
    )

    # Results: mean-subtracted
    print("\n--- B.2: Mean-subtracted hidden states (prompt-only) ---")
    # Use prompt-only background
    bg_prompt = extractor.extract_prompt_only(
        ["Map: [No interesting tiles in view - all background]\nInventory:\nHealth: 9.0\nFood: 9\nDrink: 9\nEnergy: 9\nDirection: right"]
    )[0]
    print(f"  Prompt-only background norm: {np.linalg.norm(bg_prompt):.2f}")

    hs_orig_prompt_ms = mean_subtract(hs_orig_prompt, bg_prompt)
    hs_repeats_prompt_ms = [mean_subtract(h, bg_prompt) for h in hs_repeats_prompt]
    hs_aug_prompt_ms = {k: mean_subtract(v, bg_prompt) for k, v in hs_aug_prompt.items()}
    hs_junk_prompt_ms = mean_subtract(hs_junk_prompt, bg_prompt)

    cos_b2, l2_b2 = run_comparison_suite(
        hs_orig_prompt_ms, hs_repeats_prompt_ms, hs_aug_prompt_ms, hs_junk_prompt_ms,
        "Prompt-Only (mean-subtracted)"
    )

    # =========================================================================
    # EXPERIMENT C: Last Generated Token (post-CoT reasoning state)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT C: Last Generated Token (post-CoT state)")
    print("=" * 70)

    print("\n[7/9] Extracting last-generated-token hidden states (greedy)...")

    print("  Original observations...")
    t0 = time.time()
    hs_orig_last, texts_orig_last = extractor.extract_last_generated_token(observations, temperature=0.0)
    print(f"    Done in {time.time() - t0:.1f}s")

    print(f"\n  [DEBUG] Last-token sanity check:")
    for i in range(len(observations)):
        norm = np.linalg.norm(hs_orig_last[i])
        vals = hs_orig_last[i][:5]
        print(f"    Obs {i}: norm={norm:.4f}, first5={vals}")
    if len(observations) >= 2:
        cs = cosine_similarity(hs_orig_last[0], hs_orig_last[1])
        l2 = l2_distance(hs_orig_last[0], hs_orig_last[1])
        print(f"    Obs0 vs Obs1: cos={cs:.6f}, l2={l2:.6f}")

    # Repeats (greedy = deterministic)
    print(f"\n  Repeated inference ({args.num_repeats} repeats)...")
    hs_repeats_last = []
    for rep in range(args.num_repeats):
        t0 = time.time()
        hs_rep, _ = extractor.extract_last_generated_token(observations, temperature=0.0)
        hs_repeats_last.append(hs_rep)
        print(f"    Repeat {rep+1}/{args.num_repeats} done in {time.time() - t0:.1f}s")

    # Augmented 
    hs_aug_last = {}
    for aug_name, aug_list in augmented_obs.items():
        print(f"  {aug_name}...")
        t0 = time.time()
        hs_aug, _ = extractor.extract_last_generated_token(aug_list, temperature=0.0)
        hs_aug_last[aug_name] = hs_aug
        print(f"    Done in {time.time() - t0:.1f}s")

    # Junk
    print("  Junk text...")
    hs_junk_last, _ = extractor.extract_last_generated_token(junk_observations, temperature=0.0)

    # Results: raw
    print("\n--- C.1: Raw hidden states (last generated token, greedy) ---")
    cos_c1, l2_c1 = run_comparison_suite(
        hs_orig_last, hs_repeats_last, hs_aug_last, hs_junk_last,
        "Last-Generated-Token Greedy (raw)"
    )

    # Results: mean-subtracted
    print("\n--- C.2: Mean-subtracted hidden states (last generated token) ---")
    hs_orig_last_ms = mean_subtract(hs_orig_last, background)
    hs_repeats_last_ms = [mean_subtract(h, background) for h in hs_repeats_last]
    hs_aug_last_ms = {k: mean_subtract(v, background) for k, v in hs_aug_last.items()}
    hs_junk_last_ms = mean_subtract(hs_junk_last, background)

    cos_c2, l2_c2 = run_comparison_suite(
        hs_orig_last_ms, hs_repeats_last_ms, hs_aug_last_ms, hs_junk_last_ms,
        "Last-Generated-Token Greedy (mean-subtracted)"
    )

    # =========================================================================
    # EXPERIMENT D: Layer Sweep (which layer has best representations?)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT D: Layer Sweep (Prompt-Only, Mean-Subtracted)")
    print("=" * 70)
    print("""
  This experiment tests which transformer layer produces hidden states
  that best discriminate between different observations. For each layer,
  we compute three key metrics:

    1. Aug-Diff Gap:  cos(augmented, original) - cos(different_obs, original)
       Positive = same-meaning obs are closer than different-meaning obs. 
       This is the PRIMARY quality metric — a good representation should
       map meaning-equivalent inputs to nearby vectors.

    2. Aug-Junk Gap:  cos(augmented, original) - cos(junk, original)
       Positive = real content is far from random text.
       This measures whether the representation is content-sensitive at all.

    3. Augmented cosine: cos(augmented, original)
       How stable the representation is under meaning-preserving transforms.
       Should be high (~0.99) for a good representation.

  Qwen3-4B has 36 transformer layers. Layer 0 = embedding output.
  Prior work suggests layers 28-32 may be better for reasoning.
""")

    # We can extract ALL layers in a single forward pass (output_hidden_states=True
    # already returns all layers). We just need to do it for originals, augmented,
    # and junk, then compute metrics per layer.
    
    # Layers to test: every 4th + specifically 28-32
    test_layers = sorted(set([4, 8, 12, 16, 20, 24, 28, 29, 30, 31, 32, 34, 36]))
    
    print(f"  Testing layers: {test_layers}")
    print(f"  Extracting all layers with single forward pass per observation...")
    
    def extract_all_layers(observations_list, batch_size=1):
        """Extract hidden states from ALL layers in a single forward pass.
        Returns dict: layer_idx -> (N, hidden_size) array."""
        # Accumulate per-layer results
        layer_results = {l: [] for l in test_layers}
        
        for batch_start in range(0, len(observations_list), batch_size):
            batch_obs = observations_list[batch_start:batch_start + batch_size]
            prompts = [create_prompt(obs, extractor.tokenizer) for obs in batch_obs]
            
            inputs = extractor.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=False,
            ).to(extractor.model.device)
            
            with torch.no_grad():
                outputs = extractor.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # outputs.hidden_states: tuple of (num_layers+1) tensors
            for layer_idx in test_layers:
                hs = outputs.hidden_states[layer_idx][:, -1, :].cpu().numpy().astype(np.float32)
                layer_results[layer_idx].append(hs)
        
        return {l: np.concatenate(v, axis=0) for l, v in layer_results.items()}
    
    t0 = time.time()
    orig_layers = extract_all_layers(observations)
    print(f"    Originals: {time.time() - t0:.1f}s")
    
    # Extract backgrounds per layer for mean subtraction
    t0 = time.time()
    bg_layers = extract_all_layers([
        "Map: [No interesting tiles in view - all background]\nInventory:\nHealth: 9.0\nFood: 9\nDrink: 9\nEnergy: 9\nDirection: right"
    ])
    print(f"    Background: {time.time() - t0:.1f}s")
    
    # Augmented (use first augmentation for speed)
    aug_layers_all = {}
    for aug_name, aug_list in augmented_obs.items():
        t0 = time.time()
        aug_layers_all[aug_name] = extract_all_layers(aug_list)
        print(f"    {aug_name}: {time.time() - t0:.1f}s")
    
    t0 = time.time()
    junk_layers = extract_all_layers(junk_observations)
    print(f"    Junk: {time.time() - t0:.1f}s")
    
    # Compute metrics per layer
    print(f"\n  {'Layer':>5}  {'Aug cos':>8}  {'Diff cos':>8}  {'Junk cos':>8}  {'Aug-Diff':>8}  {'Aug-Junk':>8}  {'Verdict':>20}")
    print("  " + "-" * 85)
    
    layer_summary = {}
    best_gap = -999
    best_layer = -1
    
    for layer_idx in test_layers:
        # Mean-subtract
        bg_vec = bg_layers[layer_idx][0]
        orig_ms = mean_subtract(orig_layers[layer_idx], bg_vec)
        junk_ms = mean_subtract(junk_layers[layer_idx], bg_vec)
        
        # Augmented cosines (average across all augmentations)
        all_aug_cos = []
        for aug_name in augmented_obs:
            aug_ms = mean_subtract(aug_layers_all[aug_name][layer_idx], bg_vec)
            m = compute_all_metrics(orig_ms, aug_ms)
            all_aug_cos.extend(m["cos"])
        aug_mean = np.mean(all_aug_cos)
        
        # Different observations
        cross = compute_cross_metrics(orig_ms)
        diff_mean = np.mean(cross["cos"])
        
        # Junk
        junk_m = compute_all_metrics(orig_ms, junk_ms)
        junk_mean = np.mean(junk_m["cos"])
        
        gap_ad = aug_mean - diff_mean
        gap_aj = aug_mean - junk_mean
        
        if gap_ad > 0.02:
            verdict = "✓ discriminates"
        elif gap_ad > 0:
            verdict = "~ weak"
        else:
            verdict = "✗ none"
        
        print(f"  {layer_idx:>5}  {aug_mean:>8.4f}  {diff_mean:>8.4f}  {junk_mean:>8.4f}  {gap_ad:>+8.4f}  {gap_aj:>+8.4f}  {verdict:>20}")
        
        layer_summary[layer_idx] = {
            "aug_cos": aug_mean,
            "diff_cos": diff_mean,
            "junk_cos": junk_mean,
            "gap_aug_diff": gap_ad,
            "gap_aug_junk": gap_aj,
        }
        
        if gap_ad > best_gap:
            best_gap = gap_ad
            best_layer = layer_idx
    
    print(f"\n  Best layer for discrimination: Layer {best_layer} (Aug-Diff gap: {best_gap:+.4f})")

    # =========================================================================
    # EXPERIMENT E: PCA Visualization
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT E: PCA Visualization")
    print("=" * 70)

    print("\n[9/10] Generating PCA plots...")
    results_dir = PROJECT_ROOT / "analysis" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # PCA on prompt-only mean-subtracted (best signal extraction)
    pca_dict_prompt_ms = {"Original": hs_orig_prompt_ms}
    for aug_name, hs in hs_aug_prompt_ms.items():
        pca_dict_prompt_ms[aug_name] = hs
    pca_dict_prompt_ms["Junk"] = hs_junk_prompt_ms
    save_pca_plot(
        pca_dict_prompt_ms,
        str(results_dir / f"pca_prompt_meansub_{timestamp}.png"),
        "PCA: Prompt-Only, Mean-Subtracted"
    )

    # PCA on greedy generative mean-subtracted
    pca_dict_greedy_ms = {"Original": hs_orig_greedy_ms}
    for aug_name, hs in hs_aug_greedy_ms.items():
        pca_dict_greedy_ms[aug_name] = hs
    pca_dict_greedy_ms["Junk"] = hs_junk_greedy_ms
    save_pca_plot(
        pca_dict_greedy_ms,
        str(results_dir / f"pca_greedy_meansub_{timestamp}.png"),
        "PCA: Greedy Generative, Mean-Subtracted"
    )

    # PCA on raw prompt-only (for comparison)
    pca_dict_prompt_raw = {"Original": hs_orig_prompt}
    for aug_name, hs in hs_aug_prompt.items():
        pca_dict_prompt_raw[aug_name] = hs
    pca_dict_prompt_raw["Junk"] = hs_junk_prompt
    save_pca_plot(
        pca_dict_prompt_raw,
        str(results_dir / f"pca_prompt_raw_{timestamp}.png"),
        "PCA: Prompt-Only, Raw"
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  [10/10] SUMMARY")
    print("=" * 70)

    def summarize(cos_results: Dict, label: str):
        aug_cos = []
        for k, v in cos_results.items():
            if k.startswith("Augmented"):
                aug_cos.extend(v)
        diff_cos = cos_results.get("Different observations", [0])
        junk_cos = cos_results.get("Random text vs real", [0])
        repeat_cos = cos_results.get("Repeated inference (noise floor)", [0])

        print(f"\n  {label}:")
        print(f"    Repeated inference:  {np.mean(repeat_cos):.4f} (should be ~1.0 if deterministic)")
        print(f"    Augmented (same obs): {np.mean(aug_cos):.4f}")
        print(f"    Different obs:        {np.mean(diff_cos):.4f}")
        print(f"    Random text:          {np.mean(junk_cos):.4f}")
        
        # Discrimination ratio: how much gap between same-obs and different-obs
        aug_mean = np.mean(aug_cos)
        diff_mean = np.mean(diff_cos)
        junk_mean = np.mean(junk_cos)
        
        if aug_mean > 0:
            print(f"    Gap (aug vs diff):    {aug_mean - diff_mean:+.4f}")
            print(f"    Gap (aug vs junk):    {aug_mean - junk_mean:+.4f}")
        
        # Verdict
        if np.mean(repeat_cos) > 0.999:
            det = "✓ Deterministic"
        else:
            det = f"✗ Stochastic (noise)"
        
        if aug_mean - diff_mean > 0.05:
            signal = "✓ Meaningful discrimination"
        elif aug_mean - diff_mean > 0.02:
            signal = "~ Weak discrimination"
        else:
            signal = "✗ No discrimination"
        
        if aug_mean - junk_mean > 0.1:
            content = "✓ Content-sensitive"
        elif aug_mean - junk_mean > 0.03:
            content = "~ Weakly content-sensitive"
        else:
            content = "✗ Content-insensitive"
        
        print(f"    Verdict: {det} | {signal} | {content}")

    summarize(cos_a1, "A.1 Generative Greedy (raw)")
    summarize(cos_a2, "A.2 Generative Greedy (mean-subtracted)")
    summarize(cos_b1, "B.1 Prompt-Only (raw)")
    summarize(cos_b2, "B.2 Prompt-Only (mean-subtracted)")
    summarize(cos_c1, "C.1 Last-Generated-Token Greedy (raw)")
    summarize(cos_c2, "C.2 Last-Generated-Token Greedy (mean-subtracted)")

    # Save all results
    metrics_path = results_dir / f"noise_analysis_v2_{timestamp}.json"
    all_metrics = {
        "A1_generative_greedy_raw": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]} for k, v in cos_a1.items()},
        "A2_generative_greedy_meansub": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]} for k, v in cos_a2.items()},
        "B1_prompt_only_raw": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]} for k, v in cos_b1.items()},
        "B2_prompt_only_meansub": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]} for k, v in cos_b2.items()},
        "C1_last_token_greedy_raw": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]} for k, v in cos_c1.items()},
        "C2_last_token_greedy_meansub": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]} for k, v in cos_c2.items()},
        "D_layer_sweep": {str(k): v for k, v in layer_summary.items()},
        "D_best_layer": best_layer,
        "config": {
            "model_id": args.model_id,
            "tokens_generated": args.tokens_generated,
            "num_obs": args.num_obs,
            "num_repeats": args.num_repeats,
            "hidden_size": extractor.hidden_size,
            "num_layers": 36,
            "background_norm_generative": float(bg_norm),
            "background_norm_prompt": float(np.linalg.norm(bg_prompt)),
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  All metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidden State Noise Analysis v2")
    parser.add_argument("--num_obs", type=int, default=4)
    parser.add_argument("--num_repeats", type=int, default=3)
    parser.add_argument("--tokens_generated", type=int, default=256)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    args = parser.parse_args()
    run_analysis(args)
