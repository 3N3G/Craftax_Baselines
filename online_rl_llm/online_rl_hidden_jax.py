#!/usr/bin/env python3
"""
Online RL with LLM Hidden States - Optimized JAX Implementation

Two-phase architecture for maximum performance:
- Phase A (Python): Text rendering + vLLM hidden state extraction every skip_n steps
- Phase B (JIT-compiled): Run env steps + PPO update via jax.lax.scan

Verification modes:
- --no-llm: Uses ActorCritic (no hidden state), should match ppo.py exactly
- --skip-n 1: LLM every step, matches online_rl_hidden.py behavior
- --skip-n N: LLM every N steps for speed/quality tradeoff
"""

import argparse
import os
import sys

# GPU memory sharing: JAX (XLA) and vLLM (PyTorch/CUDA) can coexist on the
# same GPU. Key settings to avoid conflicts:
#   - Disable JAX CUDA command buffers (CUDA graphs): they share a limited pool
#     with vLLM's graphs and cause "command buffer OOM" errors at instantiation.
#   - Don't preallocate: let JAX allocate on demand, vLLM already holds 60%.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"  # JAX gets 30%, vLLM gets 60%
# Disable CUDA command buffers so JAX doesn't compete with vLLM's CUDA graphs.
# Without this, JAX tries to instantiate CUDA graphs that OOM against vLLM's pool.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import time
from typing import Dict, NamedTuple, Optional, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import wandb
from craftax.craftax.constants import Achievement
from craftax.craftax_env import make_craftax_env_from_name

# Import wrappers (same as ppo.py)
from utils.wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper

# Import models from shared module (same as ppo.py uses)
from models.actor_critic import ActorCritic, ActorCriticAug

# Import text processing and vLLM interface
from utils.llm_prompts import filter_text_obs
from utils.llm_extractor import VLLMHiddenStateExtractor
import requests


# =============================================================================
# Configuration (matches ppo.py defaults)
# =============================================================================

@dataclass
class Config:
    # Environment
    ENV_NAME: str = "Craftax-Symbolic-v1"

    # LLM
    MODEL_ID: str = "Qwen/Qwen3-4B"
    HIDDEN_SIZE: int = 2560

    # Policy network
    LAYER_SIZE: int = 512

    # PPO hyperparameters (SAME as ppo.py defaults)
    LR: float = 2e-4
    ANNEAL_LR: bool = True
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.8
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 1.0
    NUM_STEPS: int = 64
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 8

    # WandB
    WANDB_PROJECT: str = "craftax-online-rl-llm"
    WANDB_ENTITY: str = "iris-sobolmark"


# =============================================================================
# Transition storage for PPO
# =============================================================================

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict


class TransitionAug(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    hidden_state: jnp.ndarray
    info: dict


# =============================================================================
# Text Observation Processing
# =============================================================================

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
# LLM Hidden State Manager
# =============================================================================

class LLMHiddenStateManager:
    def __init__(self, model_id: str = Config.MODEL_ID, target_layer: int = -1, tokens_to_generate: int = 1):
        self.tokens_to_generate = tokens_to_generate
        vllm_url = "http://localhost:8000"
        try:
            resp = requests.get(f"{vllm_url}/health", timeout=2)
            if resp.status_code != 200:
                raise Exception(f"Server returned status {resp.status_code}")
        except Exception as e:
            print(f"\n❌ ERROR: vLLM server not available at {vllm_url}")
            print(f"   Error: {e}")
            print(f"\n📝 To start: bash scripts/start_vllm_hidden.sh --mode last_token")
            sys.exit(1)

        print(f"✅ vLLM server connected at {vllm_url}")
        model_name = "./configs/vllm_hidden_qwen4b"
        extracted_layers = [8, 16, 24, 35]
        layer_index = -1 if target_layer == -1 else (extracted_layers.index(target_layer) if target_layer in extracted_layers else -1)

        self.llm = VLLMHiddenStateExtractor(server_url=vllm_url, model_name=model_name, model_id=model_id, target_layer=layer_index)
        self.hidden_size = self.llm.hidden_size
        print(f"   Hidden size: {self.hidden_size}")

    def extract(self, env_states, num_envs: int) -> Tuple[jnp.ndarray, Dict]:
        t_start = time.perf_counter()
        text_observations = []
        for i in range(num_envs):
            single_state = jax.tree.map(lambda x: x[i], env_states)
            raw_text = render_craftax_text_swapped(single_state)
            filtered_text = filter_text_obs(raw_text)
            text_observations.append(filtered_text)
        t_text = time.perf_counter() - t_start

        t_llm_start = time.perf_counter()
        if self.tokens_to_generate == 1:
            hidden_np, llm_metrics = self.llm.extract_hidden_states_no_cot(text_observations)
        else:
            hidden_np, _, llm_metrics = self.llm.extract_hidden_states(text_observations, batch_size=min(32, len(text_observations)), max_new_tokens=self.tokens_to_generate)
        t_llm = time.perf_counter() - t_llm_start

        return jnp.asarray(hidden_np), {"timing/text_render_ms": t_text * 1000, "timing/llm_inference_ms": t_llm * 1000, **llm_metrics}


# =============================================================================
# PPO Training - No LLM Mode (matches ppo.py exactly)
# =============================================================================

def make_train_no_llm(config, network, env, env_params):
    """Create JIT-compiled training functions for no-LLM mode."""

    @jax.jit
    def _env_step(carry, unused):
        train_state, env_state, last_obs, rng = carry
        rng, _rng = jax.random.split(rng)
        pi, value = network.apply(train_state.params, last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        transition = Transition(done=done, action=action, value=value, reward=reward, log_prob=log_prob, obs=last_obs, info=info)
        return (train_state, env_state, obsv, rng), transition

    @jax.jit
    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + config["GAMMA"] * next_value * (1 - done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
            return (gae, value), gae
        _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
        return advantages, advantages + traj_batch.value

    @jax.jit
    def _update_epoch(update_state, unused):
        def _update_minibatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info
            def _loss_fn(params, traj_batch, gae, targets):
                pi, value = network.apply(params, traj_batch.obs)
                log_prob = pi.log_prob(traj_batch.action)
                value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_loss = 0.5 * jnp.maximum(jnp.square(value - targets), jnp.square(value_pred_clipped - targets)).mean()
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor = -jnp.minimum(ratio * gae, jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae).mean()
                entropy = pi.entropy().mean()
                return loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy, (value_loss, loss_actor, entropy)
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
            return train_state.apply_gradients(grads=grads), total_loss

        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
        permutation = jax.random.permutation(_rng, batch_size)
        batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), (traj_batch, advantages, targets))
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree.map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled_batch)
        train_state, losses = jax.lax.scan(_update_minibatch, train_state, minibatches)
        return (train_state, traj_batch, advantages, targets, rng), losses

    @jax.jit
    def _ppo_update(train_state, traj_batch, last_obs, rng):
        _, last_val = network.apply(train_state.params, last_obs)
        advantages, targets = _calculate_gae(traj_batch, last_val)
        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
        return update_state[0], update_state[-1], loss_info

    return _env_step, _ppo_update


# =============================================================================
# PPO Training - With LLM Hidden States
# =============================================================================

def make_train_with_llm(config, network, env, env_params):
    """Create JIT-compiled training functions for LLM-augmented mode."""

    @jax.jit
    def _env_step(carry, unused):
        train_state, env_state, last_obs, hidden_states, rng = carry
        rng, _rng = jax.random.split(rng)
        pi, value = network.apply(train_state.params, last_obs, hidden_states)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        transition = TransitionAug(done=done, action=action, value=value, reward=reward, log_prob=log_prob, obs=last_obs, hidden_state=hidden_states, info=info)
        return (train_state, env_state, obsv, hidden_states, rng), transition

    @jax.jit
    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + config["GAMMA"] * next_value * (1 - done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
            return (gae, value), gae
        _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
        return advantages, advantages + traj_batch.value

    @jax.jit
    def _update_epoch(update_state, unused):
        def _update_minibatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info
            def _loss_fn(params, traj_batch, gae, targets):
                pi, value = network.apply(params, traj_batch.obs, traj_batch.hidden_state)
                log_prob = pi.log_prob(traj_batch.action)
                value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_loss = 0.5 * jnp.maximum(jnp.square(value - targets), jnp.square(value_pred_clipped - targets)).mean()
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor = -jnp.minimum(ratio * gae, jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae).mean()
                entropy = pi.entropy().mean()
                return loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy, (value_loss, loss_actor, entropy)
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
            return train_state.apply_gradients(grads=grads), total_loss

        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
        permutation = jax.random.permutation(_rng, batch_size)
        batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), (traj_batch, advantages, targets))
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree.map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled_batch)
        train_state, losses = jax.lax.scan(_update_minibatch, train_state, minibatches)
        return (train_state, traj_batch, advantages, targets, rng), losses

    @jax.jit
    def _ppo_update(train_state, traj_batch, last_obs, hidden_states, rng):
        _, last_val = network.apply(train_state.params, last_obs, hidden_states)
        advantages, targets = _calculate_gae(traj_batch, last_val)
        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
        return update_state[0], update_state[-1], loss_info

    return _env_step, _ppo_update


# =============================================================================
# Training Loop - No LLM (matches ppo.py)
# =============================================================================

def run_training_no_llm(num_envs: int, total_timesteps: int, num_steps: int, use_wandb: bool, seed: int, verbose: bool) -> Dict:
    print("=" * 70)
    print("Online RL - NO LLM MODE (matches ppo.py)")
    print("=" * 70)

    config = {
        "NUM_ENVS": num_envs, "NUM_STEPS": num_steps, "NUM_MINIBATCHES": Config.NUM_MINIBATCHES,
        "UPDATE_EPOCHS": Config.UPDATE_EPOCHS, "MINIBATCH_SIZE": num_envs * num_steps // Config.NUM_MINIBATCHES,
        "NUM_UPDATES": total_timesteps // num_steps // num_envs, "LR": Config.LR, "GAMMA": Config.GAMMA,
        "GAE_LAMBDA": Config.GAE_LAMBDA, "CLIP_EPS": Config.CLIP_EPS, "ENT_COEF": Config.ENT_COEF,
        "VF_COEF": Config.VF_COEF, "MAX_GRAD_NORM": Config.MAX_GRAD_NORM,
    }

    env = make_craftax_env_from_name(Config.ENV_NAME, auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=num_envs)

    network = ActorCritic(env.action_space(env_params).n, Config.LAYER_SIZE)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
    network_params = network.init(init_rng, init_x)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    _env_step, _ppo_update = make_train_no_llm(config, network, env, env_params)

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)

    total_steps, episode_returns = 0, []
    start_time = time.perf_counter()
    last_log_time, last_log_steps = start_time, 0

    for update_idx in range(config["NUM_UPDATES"]):
        carry = (train_state, env_state, obs, rng)
        carry, traj_batch = jax.lax.scan(_env_step, carry, None, num_steps)
        train_state, env_state, obs, rng = carry
        total_steps += num_steps * num_envs

        rng, update_rng = jax.random.split(rng)
        train_state, rng, _ = _ppo_update(train_state, traj_batch, obs, update_rng)

        completed_mask = traj_batch.info["returned_episode"].flatten()
        completed_returns = traj_batch.info["returned_episode_returns"].flatten()[completed_mask]
        if len(completed_returns) > 0:
            episode_returns.extend(completed_returns.tolist())

        current_time = time.perf_counter()
        if (update_idx + 1) % 10 == 0:
            sps = (total_steps - last_log_steps) / (current_time - last_log_time)
            mean_return = np.mean(episode_returns[-100:]) if episode_returns else 0
            if verbose:
                print(f"Update {update_idx+1:4d}/{config['NUM_UPDATES']} | Steps: {total_steps:,} | SPS: {sps:,.0f} | Return: {mean_return:.1f}")
            if use_wandb:
                wandb.log({"timestep": total_steps, "perf/sps": sps, "train/episode_return": mean_return}, step=total_steps)
            last_log_time, last_log_steps = current_time, total_steps

    total_time = time.perf_counter() - start_time
    print(f"\nDone. SPS: {total_steps/total_time:,.0f}, Return: {np.mean(episode_returns[-100:]) if episode_returns else 0:.1f}")
    return {"sps": total_steps/total_time, "final_return": np.mean(episode_returns[-100:]) if episode_returns else 0}


# =============================================================================
# Training Loop - With LLM
# =============================================================================

def run_training_with_llm(num_envs: int, total_timesteps: int, skip_n: int, num_steps: int,
                          model_id: str, target_layer: int, tokens_to_generate: int,
                          use_wandb: bool, seed: int, verbose: bool) -> Dict:
    print("=" * 70)
    print(f"Online RL with LLM Hidden States (skip_n={skip_n})")
    print("=" * 70)

    config = {
        "NUM_ENVS": num_envs, "NUM_STEPS": num_steps, "NUM_MINIBATCHES": Config.NUM_MINIBATCHES,
        "UPDATE_EPOCHS": Config.UPDATE_EPOCHS, "MINIBATCH_SIZE": num_envs * num_steps // Config.NUM_MINIBATCHES,
        "NUM_UPDATES": total_timesteps // num_steps // num_envs, "LR": Config.LR, "GAMMA": Config.GAMMA,
        "GAE_LAMBDA": Config.GAE_LAMBDA, "CLIP_EPS": Config.CLIP_EPS, "ENT_COEF": Config.ENT_COEF,
        "VF_COEF": Config.VF_COEF, "MAX_GRAD_NORM": Config.MAX_GRAD_NORM,
    }

    env = make_craftax_env_from_name(Config.ENV_NAME, auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=num_envs)

    llm_manager = LLMHiddenStateManager(model_id=model_id, target_layer=target_layer, tokens_to_generate=tokens_to_generate)

    network = ActorCriticAug(action_dim=env.action_space(env_params).n, layer_width=Config.LAYER_SIZE, hidden_state_dim=llm_manager.hidden_size)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    obs_dim = env.observation_space(env_params).shape[0]
    network_params = network.init(init_rng, jnp.zeros((1, obs_dim)), jnp.zeros((1, llm_manager.hidden_size)))

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    print("Creating JIT-compiled training functions...", flush=True)
    _env_step, _ppo_update = make_train_with_llm(config, network, env, env_params)
    print("  Done.", flush=True)

    print("Resetting environment...", flush=True)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)
    hidden_states = jnp.zeros((num_envs, llm_manager.hidden_size))
    print("  Done.", flush=True)

    total_steps, llm_calls, episode_returns = 0, 0, []
    steps_since_llm = skip_n  # Force LLM on first iter
    start_time = time.perf_counter()
    print("Starting training loop...", flush=True)
    last_log_time, last_log_steps = start_time, 0

    for update_idx in range(config["NUM_UPDATES"]):
        llm_metrics = {}
        steps_collected = 0
        all_transitions = []

        while steps_collected < num_steps:
            if steps_since_llm >= skip_n:
                hidden_states, llm_metrics = llm_manager.extract(env_state.env_state, num_envs)
                steps_since_llm = 0
                llm_calls += 1

            steps_this_chunk = min(skip_n - steps_since_llm, num_steps - steps_collected)
            steps_this_chunk = max(1, steps_this_chunk)

            carry = (train_state, env_state, obs, hidden_states, rng)
            if update_idx == 0 and steps_collected == 0:
                print(f"  First scan ({steps_this_chunk} steps)...", flush=True)
            carry, traj_chunk = jax.lax.scan(_env_step, carry, None, steps_this_chunk)
            if update_idx == 0 and steps_collected == 0:
                print(f"  First scan complete.", flush=True)
            train_state, env_state, obs, hidden_states, rng = carry

            all_transitions.append(traj_chunk)
            steps_collected += steps_this_chunk
            steps_since_llm += steps_this_chunk

        traj_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *all_transitions)
        total_steps += num_steps * num_envs

        rng, update_rng = jax.random.split(rng)
        train_state, rng, _ = _ppo_update(train_state, traj_batch, obs, hidden_states, update_rng)

        completed_mask = traj_batch.info["returned_episode"].flatten()
        completed_returns = traj_batch.info["returned_episode_returns"].flatten()[completed_mask]
        if len(completed_returns) > 0:
            episode_returns.extend(completed_returns.tolist())

        current_time = time.perf_counter()
        if (update_idx + 1) % 10 == 0:
            sps = (total_steps - last_log_steps) / (current_time - last_log_time)
            mean_return = np.mean(episode_returns[-100:]) if episode_returns else 0
            if verbose:
                print(f"Update {update_idx+1:4d}/{config['NUM_UPDATES']} | Steps: {total_steps:,} | SPS: {sps:,.0f} | Return: {mean_return:.1f} | LLM: {llm_calls}")
            if use_wandb:
                wandb.log({"timestep": total_steps, "perf/sps": sps, "perf/llm_calls": llm_calls, "train/episode_return": mean_return, **llm_metrics}, step=total_steps)
            last_log_time, last_log_steps = current_time, total_steps

    total_time = time.perf_counter() - start_time
    print(f"\nDone. SPS: {total_steps/total_time:,.0f}, LLM calls: {llm_calls}, Return: {np.mean(episode_returns[-100:]) if episode_returns else 0:.1f}")
    return {"sps": total_steps/total_time, "llm_calls": llm_calls, "final_return": np.mean(episode_returns[-100:]) if episode_returns else 0}


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Online RL with LLM hidden states (Optimized JAX)")
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--timesteps", type=lambda x: int(float(x)), default=1e6)
    parser.add_argument("--skip-n", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (matches ppo.py)")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--model", type=str, default=Config.MODEL_ID)
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=Config.WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=Config.WANDB_ENTITY)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_wandb = args.use_wandb and not args.no_wandb
    if use_wandb:
        mode_str = "no-llm" if args.no_llm else f"skip{args.skip_n}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=f"online-jax-{args.envs}env-{mode_str}", config=vars(args))

    if args.no_llm:
        results = run_training_no_llm(args.envs, args.timesteps, args.num_steps, use_wandb, args.seed, not args.quiet)
    else:
        results = run_training_with_llm(args.envs, args.timesteps, args.skip_n, args.num_steps, args.model, args.layer, args.tokens, use_wandb, args.seed, not args.quiet)

    if use_wandb:
        wandb.finish()
    print("\n✅ Done!")
    return results


if __name__ == "__main__":
    main()
