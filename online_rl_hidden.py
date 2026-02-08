#!/usr/bin/env python3
"""
Online RL with LLM Hidden States

Uses HuggingFace + Flash Attention to extract hidden states from LLM reasoning,
then feeds these into a policy head (like awr_aug.py architecture).

Key features:
- Modular hidden state extraction with optional step-skipping (skip_n parameter)
- Policy head takes hidden states as input (not parsing LLM text output)
- WandB logging with hidden state diagnostics
- Compatible with pre-trained AWR models

Usage:
    python online_rl_hidden.py --envs 8 --steps 100 --skip-n 1
    python online_rl_hidden.py --envs 8 --steps 100 --skip-n 4  # Reuse hidden states every 4 steps
"""

import argparse
import os
import re
import time
import gc
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from craftax.craftax.constants import Achievement


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # LLM
    MODEL_ID: str = "Qwen/Qwen3-4B-Thinking-2507"
    TOKENS_GENERATED: int = 256
    MAX_PROMPT_LEN: int = 2048
    HIDDEN_SIZE: int = 2560  # Qwen3-4B hidden size
    
    # Policy
    ACTION_DIM: int = 43
    LAYER_WIDTH: int = 512
    
    # Training
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # WandB
    WANDB_PROJECT: str = "craftax-online-rl-llm"
    WANDB_ENTITY: str = "iris-sobolmark"


# =============================================================================
# Observation Processing (from llm_play_harnessed.py)
# =============================================================================

BACKGROUND_TILES = {"grass", "sand", "gravel", "fire grass", "ice grass", "fire_grass", "ice_grass"}

SYSTEM_PROMPT = """You are playing Craftax.

Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. The player can move in the four cardinal directions and can interact.

The coordinate system is (Row, Column). Everything is relative to your current position.
- Negative Row is UP. Positive Row is DOWN.
- Negative Column is LEFT. Positive Column is RIGHT.
- (0, 0) is your current position.

Actions available: 
0:NOOP, 1:LEFT, 2:RIGHT, 3:UP, 4:DOWN, 5:DO (interact/mine/attack), 6:SLEEP, 7:PLACE_STONE,
8:PLACE_TABLE, 9:PLACE_FURNACE, 10:PLACE_PLANT, 11:MAKE_WOOD_PICKAXE, 12:MAKE_STONE_PICKAXE,
13:MAKE_IRON_PICKAXE, 14:MAKE_WOOD_SWORD, 15:MAKE_STONE_SWORD, 16:MAKE_IRON_SWORD, 17:REST,
18:DESCEND, 19:ASCEND, 20:MAKE_DIAMOND_PICKAXE, 21:MAKE_DIAMOND_SWORD, 22:MAKE_IRON_ARMOUR,
23:MAKE_DIAMOND_ARMOUR, 24:SHOOT_ARROW, 25:MAKE_ARROW, 26:CAST_FIREBALL, 27:CAST_ICEBALL,
28:PLACE_TORCH, 29-34:DRINK_POTION, 35:READ_BOOK, 36:ENCHANT_SWORD, 37:ENCHANT_ARMOUR, 
38:MAKE_TORCH, 39-41:LEVEL_UP, 42:ENCHANT_BOW

Analyze the game state and decide on the best action. Think step by step.
"""


def filter_text_obs(text_obs: str) -> str:
    """Filter out background tiles from observations."""
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
                    filtered_lines.append("Map: [No interesting tiles]")
                if stripped:
                    filtered_lines.append(line)
                continue
        
        if stripped:
            filtered_lines.append(line)
    
    if in_map_section:
        if interesting_tiles:
            filtered_lines.append(f"Map (interesting tiles only): {', '.join(interesting_tiles)}")
        else:
            filtered_lines.append("Map: [No interesting tiles]")
    
    return '\n'.join(filtered_lines)


def render_craftax_text_swapped(state):
    """Render text with Row,Col coordinates."""
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
# Hidden State Diagnostics
# =============================================================================

def compute_hidden_state_diagnostics(hidden_states: np.ndarray) -> Dict[str, float]:
    """
    Compute diagnostics for hidden state quality and diversity.
    
    Args:
        hidden_states: (batch, hidden_dim) pooled hidden states
        
    Returns:
        Dictionary of diagnostic metrics
    """
    if len(hidden_states) == 0:
        return {}
    
    # Basic statistics
    metrics = {
        "hidden/mean": float(np.mean(hidden_states)),
        "hidden/std": float(np.std(hidden_states)),
        "hidden/min": float(np.min(hidden_states)),
        "hidden/max": float(np.max(hidden_states)),
        "hidden/norm_mean": float(np.mean(np.linalg.norm(hidden_states, axis=1))),
    }
    
    # Per-dimension statistics (variance across batch for each dim)
    per_dim_var = np.var(hidden_states, axis=0)
    metrics["hidden/dim_var_mean"] = float(np.mean(per_dim_var))
    metrics["hidden/dim_var_std"] = float(np.std(per_dim_var))
    metrics["hidden/active_dims"] = int(np.sum(per_dim_var > 0.01))  # Dims with variance
    
    # Diversity: pairwise cosine similarity
    if len(hidden_states) >= 2:
        # Normalize
        norms = np.linalg.norm(hidden_states, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        normalized = hidden_states / norms
        
        # Compute pairwise cosine similarities (sample if too many)
        n = len(normalized)
        if n > 32:
            # Sample subset for efficiency
            idx = np.random.choice(n, 32, replace=False)
            normalized = normalized[idx]
        
        cos_sim = normalized @ normalized.T
        # Get upper triangle (exclude diagonal)
        upper_idx = np.triu_indices(len(cos_sim), k=1)
        pairwise_cos = cos_sim[upper_idx]
        
        metrics["hidden/diversity_mean_cos_sim"] = float(np.mean(pairwise_cos))
        metrics["hidden/diversity_std_cos_sim"] = float(np.std(pairwise_cos))
        metrics["hidden/diversity_min_cos_sim"] = float(np.min(pairwise_cos))
        metrics["hidden/diversity_max_cos_sim"] = float(np.max(pairwise_cos))
    
    return metrics


# =============================================================================
# LLM Hidden State Extractor
# =============================================================================

class LLMHiddenStateExtractor:
    """
    Extracts hidden states from LLM reasoning using HuggingFace + Flash Attention.
    Generates text and returns pooled hidden states.
    """
    
    def __init__(
        self,
        model_id: str = Config.MODEL_ID,
        tokens_generated: int = Config.TOKENS_GENERATED,
        device: str = Config.DEVICE,
    ):
        self.model_id = model_id
        self.tokens_generated = tokens_generated
        self.device = device
        
        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"  # Required for batched generation with decoder-only models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model with Flash Attention 2...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            self.flash_enabled = True
        except Exception as e:
            print(f"Flash Attention not available: {e}, using default")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.flash_enabled = False
        
        self.hidden_size = self.model.config.hidden_size
        print(f"Model loaded (flash_attention={self.flash_enabled}, hidden_size={self.hidden_size})")
        
        # Metrics
        self.total_samples = 0
        self.total_time = 0.0
        self.batch_times = []
    
    def format_prompt(self, observation: str) -> str:
        """Format observation into chat prompt."""
        user_content = (
            f"CURRENT GAME STATE:\n{observation}\n\n"
            f"You are at (0,0). Think step by step about what action to take."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def extract_hidden_states(
        self, 
        observations: List[str],
        batch_size: int = 8,
    ) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """
        Generate text and extract hidden states.
        
        Args:
            observations: List of filtered text observations
            batch_size: Batch size for processing
            
        Returns:
            hidden_states: (N, hidden_size) pooled hidden states
            generated_texts: List of generated text strings
            metrics: Timing and diagnostic metrics
        """
        if not observations:
            return np.array([]), [], {}
        
        start_time = time.perf_counter()
        
        all_hidden = []
        all_texts = []
        batch_latencies = []
        
        for batch_idx in range(0, len(observations), batch_size):
            batch_start = time.perf_counter()
            batch_obs = observations[batch_idx:batch_idx + batch_size]
            prompts = [self.format_prompt(obs) for obs in batch_obs]
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=Config.MAX_PROMPT_LEN,
            ).to(self.model.device)
            
            prompt_len = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.tokens_generated,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample=True,
                    temperature=0.7,
                )
            
            # Extract hidden states from last layer
            last_layer_states = [s[-1] for s in outputs.hidden_states]
            generated_hidden = torch.cat(last_layer_states, dim=1)
            
            # Truncate/pad to tokens_generated
            seq_len = generated_hidden.shape[1]
            if seq_len > self.tokens_generated:
                generated_hidden = generated_hidden[:, :self.tokens_generated, :]
            elif seq_len < self.tokens_generated:
                padding = torch.zeros(
                    (len(batch_obs), self.tokens_generated - seq_len, self.hidden_size),
                    device=generated_hidden.device,
                    dtype=generated_hidden.dtype,
                )
                generated_hidden = torch.cat([generated_hidden, padding], dim=1)
            
            # Mean pool to (batch, hidden_size)
            pooled = generated_hidden.mean(dim=1).cpu().numpy().astype(np.float32)
            all_hidden.append(pooled)
            
            # Decode text
            generated_ids = outputs.sequences[:, prompt_len:]
            texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_texts.extend(texts)
            
            batch_latencies.append(time.perf_counter() - batch_start)
        
        elapsed = time.perf_counter() - start_time
        self.total_samples += len(observations)
        self.total_time += elapsed
        self.batch_times.extend(batch_latencies)
        
        all_hidden = np.concatenate(all_hidden, axis=0)
        
        # Compute metrics
        metrics = {
            "llm/inference_time_s": elapsed,
            "llm/samples_per_sec": len(observations) / elapsed,
            "llm/batch_latency_mean_s": np.mean(batch_latencies),
            "llm/total_samples": self.total_samples,
            "llm/total_time_s": self.total_time,
        }
        
        # Add hidden state diagnostics
        hidden_diag = compute_hidden_state_diagnostics(all_hidden)
        metrics.update(hidden_diag)
        
        return all_hidden, all_texts, metrics
    
    def get_metrics(self) -> Dict:
        return {
            "total_samples": self.total_samples,
            "total_time_s": self.total_time,
            "samples_per_sec": self.total_samples / max(0.001, self.total_time),
        }


# =============================================================================
# Policy Network (from awr_aug.py)
# =============================================================================

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class PolicyHead(nn.Module):
    """
    Policy head that takes LLM hidden states and outputs actions.
    Simplified version of ActorCriticConvAug without CNN (hidden states only).
    """
    
    def __init__(self, hidden_state_dim: int, action_dim: int, layer_width: int = 512):
        super().__init__()
        
        # Actor
        self.actor_fc1 = nn.Linear(hidden_state_dim, layer_width)
        self.actor_fc2 = nn.Linear(layer_width, layer_width)
        self.actor_fc3 = nn.Linear(layer_width, action_dim)
        
        # Critic
        self.critic_fc1 = nn.Linear(hidden_state_dim, layer_width)
        self.critic_fc2 = nn.Linear(layer_width, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        orthogonal_init(self.actor_fc1, gain=2.0)
        orthogonal_init(self.actor_fc2, gain=2.0)
        orthogonal_init(self.actor_fc3, gain=0.01)
        orthogonal_init(self.critic_fc1, gain=2.0)
        orthogonal_init(self.critic_fc2, gain=1.0)
    
    def forward(self, hidden_state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Args:
            hidden_state: (batch, hidden_dim) pooled LLM hidden states
            
        Returns:
            pi: Action distribution
            value: State value estimate
        """
        # Actor
        x = F.relu(self.actor_fc1(hidden_state))
        x = F.relu(self.actor_fc2(x))
        logits = self.actor_fc3(x)
        pi = Categorical(logits=logits)
        
        # Critic
        v = F.relu(self.critic_fc1(hidden_state))
        value = self.critic_fc2(v)
        
        return pi, value.squeeze(-1)


# =============================================================================
# Online RL Agent with Hidden State Caching
# =============================================================================

class OnlineRLAgent:
    """
    Online RL agent using LLM hidden states.
    Supports step-skipping (reuse hidden states for n steps).
    """
    
    def __init__(
        self,
        num_envs: int,
        skip_n: int = 1,  # 1 = update every step, 4 = update every 4 steps
        model_id: str = Config.MODEL_ID,
        checkpoint_path: Optional[str] = None,
    ):
        self.num_envs = num_envs
        self.skip_n = skip_n
        self.step_counter = 0
        
        # Hidden state extractor
        self.llm = LLMHiddenStateExtractor(model_id=model_id)
        
        # Policy head
        self.policy = PolicyHead(
            hidden_state_dim=self.llm.hidden_size,
            action_dim=Config.ACTION_DIM,
            layer_width=Config.LAYER_WIDTH,
        ).to(Config.DEVICE)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            self.policy.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
        
        # Cached hidden states
        self.cached_hidden = None
        self.cached_hidden_np = None
        self.cached_texts = None
        self.last_metrics = {}
    
    def get_actions(
        self, 
        observations: List[str],
        force_update: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, float]]:
        """
        Get actions from policy using LLM hidden states.
        
        Args:
            observations: List of filtered text observations
            force_update: Force LLM inference even if skip_n not reached
            
        Returns:
            actions: (N,) action indices
            values: (N,) value estimates
            reasoning: List of LLM text outputs (empty if cached)
            metrics: LLM and hidden state metrics
        """
        self.step_counter += 1
        
        # Check if we should update hidden states
        should_update = (
            force_update or 
            self.cached_hidden is None or 
            (self.step_counter % self.skip_n == 0)
        )
        
        if should_update:
            # Extract new hidden states from LLM
            hidden_np, texts, metrics = self.llm.extract_hidden_states(observations)
            self.cached_hidden_np = hidden_np
            self.cached_hidden = torch.tensor(hidden_np, dtype=torch.float32, device=Config.DEVICE)
            self.cached_texts = texts
            self.last_metrics = metrics
        else:
            metrics = {"llm/cache_hit": 1.0}
        
        # Get actions from policy head
        with torch.no_grad():
            pi, values = self.policy(self.cached_hidden)
            actions = pi.sample()
            
            # Action distribution metrics
            metrics["policy/entropy"] = pi.entropy().mean().item()
            metrics["policy/value_mean"] = values.mean().item()
            metrics["policy/value_std"] = values.std().item()
            
            # Action diversity
            action_hist = torch.bincount(actions, minlength=Config.ACTION_DIM)
            metrics["policy/action_diversity"] = (action_hist > 0).sum().item()
        
        return (
            actions.cpu().numpy(),
            values.cpu().numpy(),
            self.cached_texts if should_update else [],
            metrics,
        )
    
    def get_llm_metrics(self) -> Dict:
        return self.llm.get_metrics()


# =============================================================================
# Main Training Loop
# =============================================================================

def run_online_rl(
    num_envs: int,
    num_steps: int,
    skip_n: int = 1,
    model_id: str = Config.MODEL_ID,
    checkpoint_path: Optional[str] = None,
    use_wandb: bool = True,
    verbose: bool = True,
) -> Dict:
    """Run online RL with LLM hidden states."""
    
    print("=" * 60)
    print("Online RL with LLM Hidden States")
    print("=" * 60)
    print(f"Environments: {num_envs}")
    print(f"Steps: {num_steps}")
    print(f"Skip-N: {skip_n} (LLM inference every {skip_n} steps)")
    print(f"Model: {model_id}")
    print(f"WandB: {use_wandb}")
    print("=" * 60)
    
    from craftax.craftax_env import make_craftax_env_from_name
    import jax
    
    # Initialize agent
    print("\n[1/3] Loading agent...")
    agent = OnlineRLAgent(
        num_envs=num_envs,
        skip_n=skip_n,
        model_id=model_id,
        checkpoint_path=checkpoint_path,
    )
    
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
    
    # Tracking
    total_reward = 0.0
    episode_count = 0
    episode_rewards = []
    current_episode_rewards = [0.0] * num_envs
    step_times = []
    llm_inference_count = 0
    action_counts = np.zeros(Config.ACTION_DIM, dtype=np.int32)
    
    # Achievement tracking
    achievement_names = [a.name for a in Achievement]
    total_achievements_unlocked = set()  # Set of (env_idx, achievement_name) tuples
    achievements_this_step = []  # List of achievement names unlocked this step
    prev_achievements = [None] * num_envs  # Track previous achievement state per env
    
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        
        # Render observations
        observations = []
        for state in states:
            raw_text = render_craftax_text_swapped(state)
            filtered_text = filter_text_obs(raw_text)
            observations.append(filtered_text)
        
        # Get actions from agent
        actions, values, reasoning, metrics = agent.get_actions(observations)
        
        if reasoning:  # LLM was invoked
            llm_inference_count += 1
        
        # Track actions
        for a in actions:
            action_counts[a] += 1
        
        # Step environments
        new_states = []
        step_reward = 0.0
        
        achievements_this_step = []
        
        for i, (state, action) in enumerate(zip(states, actions)):
            rng, step_rng = jax.random.split(rngs[i])
            rngs = rngs.at[i].set(rng)
            
            _, new_state, reward, done, _ = env.step(
                step_rng, state, int(action), env_params
            )
            
            step_reward += float(reward)
            current_episode_rewards[i] += float(reward)
            
            # Track new achievements
            curr_ach = np.array(new_state.achievements)
            if prev_achievements[i] is not None:
                new_ach_mask = curr_ach & ~prev_achievements[i]
                for ach_idx in np.where(new_ach_mask)[0]:
                    ach_name = achievement_names[ach_idx]
                    achievements_this_step.append(ach_name)
                    total_achievements_unlocked.add((i, ach_name))
            prev_achievements[i] = curr_ach
            
            if done:
                episode_count += 1
                episode_rewards.append(current_episode_rewards[i])
                current_episode_rewards[i] = 0.0
                prev_achievements[i] = None  # Reset on episode end
            
            new_states.append(new_state)
        
        states = new_states
        total_reward += step_reward
        step_times.append(time.perf_counter() - step_start)
        
        # Print new achievements
        if achievements_this_step and verbose:
            print(f"  🏆 Step {step+1}: {', '.join(achievements_this_step)}")
        
        # WandB logging
        if use_wandb and (step + 1) % 10 == 0:
            # Count unique achievements across all envs
            unique_achievements = set(name for _, name in total_achievements_unlocked)
            
            log_dict = {
                "step": step + 1,
                "env/step_reward": step_reward,
                "env/total_reward": total_reward,
                "env/episodes": episode_count,
                "env/episode_reward_mean": np.mean(episode_rewards) if episode_rewards else 0,
                "achievements/total_unique": len(unique_achievements),
                "achievements/total_unlocks": len(total_achievements_unlocked),
                "perf/step_time_s": np.mean(step_times[-10:]),
                "perf/samples_per_sec": num_envs / np.mean(step_times[-10:]),
                "perf/llm_inference_count": llm_inference_count,
            }
            log_dict.update(metrics)
            wandb.log(log_dict)
        
        if verbose and (step + 1) % 10 == 0:
            avg_time = np.mean(step_times[-10:])
            sps = num_envs / avg_time
            print(f"  Step {step+1:4d}/{num_steps} | "
                  f"Reward: {step_reward:+5.1f} | "
                  f"Eps: {episode_count:3d} | "
                  f"LLM: {llm_inference_count:3d} | "
                  f"SPS: {sps:.2f}")
    
    elapsed = time.perf_counter() - start_time
    
    # Results
    print("-" * 60)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    total_samples = num_envs * num_steps
    samples_per_sec = total_samples / elapsed
    
    llm_metrics = agent.get_llm_metrics()
    
    print(f"\nPerformance:")
    print(f"  Total samples:     {total_samples:,}")
    print(f"  Total time:        {elapsed:.1f}s")
    print(f"  Samples/sec:       {samples_per_sec:.2f}")
    print(f"  LLM inferences:    {llm_inference_count}")
    print(f"  LLM samples/sec:   {llm_metrics['samples_per_sec']:.2f}")
    
    print(f"\nGame stats:")
    print(f"  Episodes:          {episode_count}")
    print(f"  Total reward:      {total_reward:.1f}")
    if episode_rewards:
        print(f"  Mean episode reward: {np.mean(episode_rewards):.2f}")
    
    print(f"\nAction distribution:")
    top_actions = np.argsort(action_counts)[-5:][::-1]
    for a in top_actions:
        print(f"  Action {a}: {action_counts[a]} ({100*action_counts[a]/action_counts.sum():.1f}%)")
    
    # Achievements summary
    unique_achievements = sorted(set(name for _, name in total_achievements_unlocked))
    print(f"\n🏆 Achievements Unlocked ({len(unique_achievements)} unique):")
    if unique_achievements:
        for ach in unique_achievements:
            print(f"  ✓ {ach}")
    else:
        print("  (none)")

    
    # Final WandB summary
    if use_wandb:
        wandb.summary["final/total_samples"] = total_samples
        wandb.summary["final/total_time_s"] = elapsed
        wandb.summary["final/samples_per_sec"] = samples_per_sec
        wandb.summary["final/llm_samples_per_sec"] = llm_metrics['samples_per_sec']
        wandb.summary["final/episodes"] = episode_count
        wandb.summary["final/total_reward"] = total_reward
        if episode_rewards:
            wandb.summary["final/mean_episode_reward"] = np.mean(episode_rewards)
    
    return {
        "samples_per_sec": samples_per_sec,
        "llm_samples_per_sec": llm_metrics['samples_per_sec'],
        "total_time": elapsed,
        "llm_inference_count": llm_inference_count,
        "total_samples": total_samples,
        "episode_count": episode_count,
        "total_reward": total_reward,
    }


def estimate_training_times(llm_sps: float, skip_n: int = 1):
    """Estimate training times for different step counts."""
    print("\n" + "=" * 60)
    print("TRAINING TIME ESTIMATES")
    print("=" * 60)
    
    # Effective SPS = LLM SPS * skip_n (when we reuse hidden states)
    effective_sps = llm_sps * skip_n
    
    print(f"LLM samples/sec: {llm_sps:.2f}")
    print(f"Skip-N factor: {skip_n}x")
    print(f"Effective samples/sec: {effective_sps:.2f}")
    
    targets = [
        (2_000, "2K"),
        (10_000, "10K"),
        (50_000, "50K"),
        (100_000, "100K"),
        (500_000, "500K"),
    ]
    
    print(f"\n{'Steps':<10} {'Time':<15} {'Status'}")
    print("-" * 40)
    
    for target, label in targets:
        hours = target / effective_sps / 3600
        if hours < 1:
            time_str = f"{hours * 60:.1f} minutes"
        elif hours < 24:
            time_str = f"{hours:.1f} hours"
        else:
            time_str = f"{hours/24:.1f} days"
        
        status = "✅" if hours < 4 else "⚠️" if hours < 24 else "❌"
        print(f"{label:<10} {time_str:<15} {status}")


def main():
    parser = argparse.ArgumentParser(description="Online RL with LLM hidden states")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--skip-n", type=int, default=1,
                        help="Reuse hidden states for N steps (1=every step, 4=every 4 steps)")
    parser.add_argument("--model", type=str, default=Config.MODEL_ID)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to policy checkpoint")
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=Config.WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=Config.WANDB_ENTITY)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    use_wandb = args.use_wandb and not args.no_wandb
    
    # Initialize WandB
    if use_wandb:
        run_name = f"online-rl-hidden-{args.envs}envs-{args.steps}steps-skip{args.skip_n}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "num_envs": args.envs,
                "num_steps": args.steps,
                "skip_n": args.skip_n,
                "model_id": args.model,
                "tokens_generated": Config.TOKENS_GENERATED,
                "hidden_size": Config.HIDDEN_SIZE,
            },
        )
        print(f"WandB initialized: {args.wandb_project}/{run_name}")
    
    results = run_online_rl(
        num_envs=args.envs,
        num_steps=args.steps,
        skip_n=args.skip_n,
        model_id=args.model,
        checkpoint_path=args.checkpoint,
        use_wandb=use_wandb,
        verbose=not args.quiet,
    )
    
    # Estimate training times
    estimate_training_times(results['llm_samples_per_sec'], args.skip_n)
    
    if use_wandb:
        wandb.finish()
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
