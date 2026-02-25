import os
import glob
import argparse
import hashlib
import json
import numpy as np
import concurrent.futures
import time
from datetime import datetime, timezone
from typing import Tuple

try:
    import wandb
except ImportError:
    wandb = None

# --- FIX: Prevent Import Hangs / Deadlocks on Clusters ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

try:
    import jax  # noqa: F401
except ImportError:
    jax = None

# ==============================================================================
# 1. Configuration
# ==============================================================================
class Config:
    # Default to the actively-growing labelling output so offline experiments can
    # start before full labelling completion.
    DATA_DIR = "/data/group_data/rl/geney/vllm_craftax_labelled_results"
    DATA_GLOB = "trajectories_batch_*.npz"

    # Model
    ACTION_DIM = 43
    LAYER_WIDTH = 512
    HIDDEN_STATE_DIM = 2560  # LLM hidden state dimension
    OBS_DIM = 1345 # Symbolic observation size (check this)
    HIDDEN_MODE = "real"  # real | zero | shuffle
    FUSION_MODE = "concat_raw"  # concat_raw | gated_proj | residual_gated
    HIDDEN_GATE_INIT_LOGIT = -4.0
    ADVANTAGE_MODE = "center"  # raw | center | standardize

    GAMMA = 0.99
    NUM_ENVS = 128
    AWR_BETA = 10.0
    AWR_MAX_WEIGHT = 20.0
    LR = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    TOTAL_STEPS = 100_000
    BATCH_SIZE = 256
    LOG_FREQ = 100
    SAVE_FREQ = 25000
    SAVE_DIR = "/data/group_data/rl/geney/checkpoints/awr_llm_augmented/"
    SEED = 42
    MAX_DATASET_GB = 80.0

    # Wandb
    WANDB_PROJECT = "craftax-offline-awr"
    WANDB_ENTITY = "iris-sobolmark"


def compute_return_to_go(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    num_envs: int,
) -> Tuple[np.ndarray, int]:
    """
    Compute return-to-go for flattened trajectories.
    If shape matches interleaved multi-env rollout, compute per-env RTG exactly.
    Otherwise, fall back to single-stream RTG.
    """
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)

    if rewards.shape[0] != dones.shape[0]:
        raise ValueError(
            f"reward/done length mismatch: {rewards.shape[0]} vs {dones.shape[0]}"
        )

    if num_envs > 1 and rewards.shape[0] % num_envs == 0:
        rewards_mat = rewards.reshape(-1, num_envs)
        dones_mat = dones.reshape(-1, num_envs)
        returns_mat = np.zeros_like(rewards_mat, dtype=np.float32)

        next_return = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(rewards_mat.shape[0])):
            next_return = rewards_mat[t] + gamma * next_return * (1.0 - dones_mat[t])
            returns_mat[t] = next_return
        truncated_streams = int(np.sum(dones_mat[-1] < 0.5))
        return returns_mat.reshape(-1), truncated_streams

    returns = np.zeros_like(rewards, dtype=np.float32)
    next_return = 0.0
    for t in reversed(range(rewards.shape[0])):
        next_return = rewards[t] + gamma * next_return * (1.0 - dones[t])
        returns[t] = next_return
    truncated_streams = int(dones[-1] < 0.5)
    return returns, truncated_streams


def apply_hidden_skip_schedule(
    hidden_state: np.ndarray,
    done: np.ndarray,
    num_envs: int,
    skip_n: int,
    reset_on_done: bool,
) -> Tuple[np.ndarray, int]:
    """
    Hold hidden vectors between refresh steps to emulate online skip_n behavior.

    Args:
        hidden_state: [N, H] hidden vectors.
        done: [N] flattened done mask aligned with hidden_state.
        num_envs: Interleaving factor in flattened stream.
        skip_n: Refresh cadence. 1 means no hold.
        reset_on_done: If True, force refresh on first step after terminal.

    Returns:
        transformed_hidden, refresh_count
    """
    if skip_n <= 1:
        return hidden_state.astype(np.float32, copy=False), int(hidden_state.shape[0])
    if hidden_state.ndim != 2:
        raise ValueError(f"Expected hidden_state shape [N,H], got {hidden_state.shape}")
    hidden_state = hidden_state.astype(np.float32, copy=False)
    done = np.asarray(done, dtype=np.float32).reshape(-1)
    n, hidden_dim = hidden_state.shape
    if done.shape[0] != n:
        raise ValueError(
            f"done/hidden length mismatch: done={done.shape[0]} hidden={n}"
        )

    # Interleaved [time, env, hidden] path (common PPO save format).
    if num_envs > 1 and (n % num_envs == 0):
        t_len = n // num_envs
        hidden_3d = hidden_state.reshape(t_len, num_envs, hidden_dim)
        done_2d = done.reshape(t_len, num_envs)
        out = np.empty_like(hidden_3d)
        last = hidden_3d[0].copy()
        steps_since_refresh = np.zeros((num_envs,), dtype=np.int32)
        out[0] = last
        refresh_count = int(num_envs)
        for t in range(1, t_len):
            must_refresh = steps_since_refresh >= (skip_n - 1)
            if reset_on_done:
                must_refresh = np.logical_or(must_refresh, done_2d[t - 1] > 0.5)
            cur = hidden_3d[t]
            # Refresh per-env where requested; otherwise hold previous hidden.
            last = np.where(must_refresh[:, None], cur, last)
            out[t] = last
            steps_since_refresh = np.where(must_refresh, 0, steps_since_refresh + 1)
            refresh_count += int(np.sum(must_refresh))
        return out.reshape(n, hidden_dim), refresh_count

    # Fallback: single stream.
    out = np.empty_like(hidden_state)
    out[0] = hidden_state[0]
    refresh_count = 1
    steps_since_refresh = 0
    for i in range(1, n):
        must_refresh = steps_since_refresh >= (skip_n - 1)
        if reset_on_done and done[i - 1] > 0.5:
            must_refresh = True
        if must_refresh:
            out[i] = hidden_state[i]
            steps_since_refresh = 0
            refresh_count += 1
        else:
            out[i] = out[i - 1]
            steps_since_refresh += 1
    return out, refresh_count


# ==============================================================================
# 2. Model Architecture (Symbolic MLP + LLM Hidden States)
# ==============================================================================
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ActorCriticAug(nn.Module):
    """
    Symbolic MLP encoder + LLM hidden state augmentation.
    Matches the JAX architecture in models/actor_critic.py
    """
    def __init__(
        self,
        obs_dim,
        action_dim,
        layer_width,
        hidden_state_dim,
        fusion_mode="concat_raw",
        hidden_gate_init_logit=-4.0,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.hidden_gate_init_logit = hidden_gate_init_logit

        # MLP Encoder for symbolic observations
        # Matches ActorCritic in models/actor_critic.py (3 layers of 512)
        self.encoder_fc1 = nn.Linear(obs_dim, layer_width)
        self.encoder_fc2 = nn.Linear(layer_width, layer_width)
        self.encoder_fc3 = nn.Linear(layer_width, layer_width)

        # Fusion
        if self.fusion_mode == "concat_raw":
            self.combined_dim = layer_width + hidden_state_dim
            self.hidden_ln = None
            self.hidden_proj = None
            self.hidden_gate_logit = None
        elif self.fusion_mode == "gated_proj":
            self.hidden_ln = nn.LayerNorm(hidden_state_dim)
            self.hidden_proj = nn.Linear(hidden_state_dim, layer_width)
            self.hidden_gate_logit = nn.Parameter(
                torch.tensor(float(hidden_gate_init_logit), dtype=torch.float32)
            )
            self.combined_dim = layer_width + layer_width
        elif self.fusion_mode == "residual_gated":
            self.hidden_ln = nn.LayerNorm(hidden_state_dim)
            self.hidden_proj = nn.Linear(hidden_state_dim, layer_width)
            self.hidden_gate_logit = nn.Parameter(
                torch.tensor(float(hidden_gate_init_logit), dtype=torch.float32)
            )
            # Residual fusion keeps actor/critic input shape identical to the
            # obs-only baseline so gate->0 cleanly recovers baseline behavior.
            self.combined_dim = layer_width
        else:
            raise ValueError(
                f"Unknown fusion_mode={fusion_mode}. "
                "Expected one of: concat_raw, gated_proj, residual_gated."
            )

        # Actor head
        self.actor_fc1 = nn.Linear(self.combined_dim, layer_width)
        self.actor_fc2 = nn.Linear(layer_width, action_dim)
        
        # Critic head
        self.critic_fc1 = nn.Linear(self.combined_dim, layer_width)
        self.critic_fc2 = nn.Linear(layer_width, layer_width)
        self.critic_fc3 = nn.Linear(layer_width, layer_width)
        self.critic_out = nn.Linear(layer_width, 1)

        self.apply_init()

    def apply_init(self):
        # Orthogonal initialization
        orthogonal_init(self.encoder_fc1, gain=np.sqrt(2))
        orthogonal_init(self.encoder_fc2, gain=np.sqrt(2))
        orthogonal_init(self.encoder_fc3, gain=np.sqrt(2))
        if self.hidden_proj is not None:
            orthogonal_init(self.hidden_proj, gain=np.sqrt(2))
        
        orthogonal_init(self.actor_fc1, gain=np.sqrt(2))
        orthogonal_init(self.actor_fc2, gain=0.01)

        orthogonal_init(self.critic_fc1, gain=np.sqrt(2))
        orthogonal_init(self.critic_fc2, gain=np.sqrt(2))
        orthogonal_init(self.critic_fc3, gain=np.sqrt(2))
        orthogonal_init(self.critic_out, gain=1.0)

    def forward(self, obs, hidden_state):
        # MLP Encoder
        x = F.tanh(self.encoder_fc1(obs))
        x = F.tanh(self.encoder_fc2(x))
        x = F.tanh(self.encoder_fc3(x))

        if self.fusion_mode == "concat_raw":
            combined = torch.cat([x, hidden_state], dim=1)
        elif self.fusion_mode == "gated_proj":
            hidden_proc = self.hidden_ln(hidden_state)
            hidden_proc = torch.tanh(self.hidden_proj(hidden_proc))
            hidden_gate = torch.sigmoid(self.hidden_gate_logit)
            combined = torch.cat([x, hidden_gate * hidden_proc], dim=1)
        elif self.fusion_mode == "residual_gated":
            hidden_proc = self.hidden_ln(hidden_state)
            hidden_proc = torch.tanh(self.hidden_proj(hidden_proc))
            hidden_gate = torch.sigmoid(self.hidden_gate_logit)
            combined = x + hidden_gate * hidden_proc
        else:
            raise ValueError(
                f"Unknown fusion_mode={self.fusion_mode}. "
                "Expected one of: concat_raw, gated_proj, residual_gated."
            )

        # Actor
        actor_x = F.tanh(self.actor_fc1(combined))
        actor_logits = self.actor_fc2(actor_x)
        pi = Categorical(logits=actor_logits)

        # Critic
        critic_x = F.tanh(self.critic_fc1(combined))
        critic_x = F.tanh(self.critic_fc2(critic_x))
        critic_x = F.tanh(self.critic_fc3(critic_x))
        value = self.critic_out(critic_x)

        return pi, value.squeeze(-1)


# ==============================================================================
# 3. Dataset Loader (Augmented with Hidden States)
# ==============================================================================
class OfflineDatasetLLMAugmented:
    def __init__(
        self,
        data_dir,
        file_pattern,
        max_files=None,
        num_envs=128,
        compute_missing_returns=True,
        max_workers=8,
        hidden_mode="real",
        hidden_skip_n=1,
        hidden_skip_reset_on_done=False,
        max_dataset_gb=80.0,
        auto_file_limit=True,
        min_rtg_quantile=0.0,
    ):
        if hidden_mode not in {"real", "zero", "shuffle"}:
            raise ValueError(
                f"Unsupported hidden_mode={hidden_mode}. Expected one of: real, zero, shuffle."
            )

        search_path = os.path.join(data_dir, file_pattern)
        files = glob.glob(search_path)
        if not files:
            raise ValueError(f"No files found at {search_path}")

        files = sorted(files)
        if max_files is not None:
            files = files[:max_files]
        print(f"Found {len(files)} LLM-labelled files.")
        print(f"Hidden mode: {hidden_mode}")
        print(
            f"Hidden skip: {hidden_skip_n} "
            f"(reset_on_done={hidden_skip_reset_on_done})"
        )

        if len(files) == 0:
            raise ValueError(f"No files found at {search_path}")

        # Find observation dimension from first readable file
        first_readable_file = None
        for f in files:
            try:
                with np.load(f, mmap_mode="r") as d:
                    obs_shape = d["obs"].shape
                    Config.OBS_DIM = np.prod(obs_shape[1:])
                    first_readable_file = f
                    print(
                        f"Observation dimension: {Config.OBS_DIM} "
                        f"(shape: {obs_shape[1:]})"
                    )
                    break
            except Exception:
                continue
        if first_readable_file is None:
            raise ValueError("No readable files with `obs` found in dataset.")

        # Count total samples
        total_samples = 0
        file_info = []
        for f in files:
            try:
                with np.load(f, mmap_mode="r") as d:
                    n = d["reward"].shape[0]
                    total_samples += n
                    file_info.append((f, n))
            except Exception as e:
                print(f"Skipping corrupt file {f}: {e}")

        if len(file_info) == 0:
            raise ValueError("No valid files remained after loading metadata.")

        def estimate_buffer_gb(sample_count: int) -> float:
            # Arrays held in memory: obs(float32), hidden_state(float32),
            # reward/done/return_to_go(float32), action(int32).
            bytes_per_sample = (
                4 * Config.OBS_DIM
                + 4 * Config.HIDDEN_STATE_DIM
                + 3 * 4
                + 4
            )
            return (sample_count * bytes_per_sample) / (1024 ** 3)

        estimated_gb = estimate_buffer_gb(total_samples)
        print(f"Estimated in-memory dataset footprint: {estimated_gb:.2f} GiB")

        if max_dataset_gb is not None and estimated_gb > max_dataset_gb:
            if auto_file_limit:
                original_files = len(file_info)
                original_samples = total_samples
                limited_file_info = []
                limited_samples = 0
                for info in file_info:
                    next_samples = limited_samples + info[1]
                    if (
                        len(limited_file_info) > 0
                        and estimate_buffer_gb(next_samples) > max_dataset_gb
                    ):
                        break
                    limited_file_info.append(info)
                    limited_samples = next_samples

                if len(limited_file_info) == 0:
                    raise MemoryError(
                        "Dataset exceeds memory budget and even one file does not fit. "
                        f"Estimated memory for first file: {estimate_buffer_gb(file_info[0][1]):.2f} GiB, "
                        f"budget: {max_dataset_gb:.2f} GiB."
                    )

                file_info = limited_file_info
                files = [f for f, _ in file_info]
                total_samples = limited_samples
                estimated_gb = estimate_buffer_gb(total_samples)
                print(
                    "WARNING: auto-limited files to stay within memory budget. "
                    f"Using {len(file_info)}/{original_files} files and "
                    f"{total_samples}/{original_samples} samples "
                    f"(estimated {estimated_gb:.2f} GiB <= {max_dataset_gb:.2f} GiB)."
                )
            else:
                raise MemoryError(
                    "Estimated dataset buffers exceed memory budget. "
                    f"Need ~{estimated_gb:.2f} GiB for {total_samples} samples, "
                    f"budget is {max_dataset_gb:.2f} GiB. "
                    "Use --max_files, increase --max_dataset_gb, or enable auto file limiting."
                )

        print(f"Allocating buffers for {total_samples} samples...")

        # Allocate arrays
        self.obs = np.zeros((total_samples, Config.OBS_DIM), dtype=np.float32)
        self.action = np.zeros((total_samples,), dtype=np.int32)
        self.reward = np.zeros((total_samples,), dtype=np.float32)
        self.done = np.zeros((total_samples,), dtype=np.float32)
        need_hidden = hidden_mode in {"real", "shuffle"}
        self.hidden_state = (
            np.zeros((total_samples, Config.HIDDEN_STATE_DIM), dtype=np.float32)
            if need_hidden
            else None
        )
        self.return_to_go = np.zeros((total_samples,), dtype=np.float32)
        self.hidden_mode = hidden_mode
        self.hidden_skip_n = int(hidden_skip_n)
        self.hidden_skip_reset_on_done = bool(hidden_skip_reset_on_done)
        self._dataset_files = [f for f, _ in file_info]

        def load_single_file(args):
            fpath, expected_n = args
            try:
                with np.load(fpath) as data:
                    # Load obs (flatten if needed)
                    raw_obs = data["obs"]
                    if len(raw_obs.shape) > 2:
                        raw_obs = raw_obs.reshape(raw_obs.shape[0], -1)
                    
                    pooled_hidden = None
                    if need_hidden:
                        # Note: hidden states from LLM might be (N, seq, dim) or (N, dim)
                        # Check shape and pool if necessary
                        raw_hidden = data["hidden_state"]
                        if len(raw_hidden.shape) == 3:
                            # (N, seq, dim) -> mean pool
                            pooled_hidden = np.mean(raw_hidden, axis=1)
                        else:
                            pooled_hidden = raw_hidden

                    if "return_to_go" in data:
                        return_to_go = data["return_to_go"].astype(np.float32)
                        computed_returns = False
                        truncated_streams = 0
                    elif compute_missing_returns and "reward" in data and "done" in data:
                        return_to_go, truncated_streams = compute_return_to_go(
                            data["reward"], data["done"], Config.GAMMA, num_envs
                        )
                        computed_returns = True
                    else:
                        raise KeyError(
                            "Missing return_to_go and cannot compute it "
                            "(require reward/done or enable --compute-missing-returns)."
                        )

                    return {
                        "obs": raw_obs.astype(np.float32),
                        "action": data["action"],
                        "reward": data["reward"].astype(np.float32),
                        "done": data["done"],
                        "hidden_state": None if pooled_hidden is None else pooled_hidden.astype(np.float32),
                        "return_to_go": return_to_go,
                        "computed_returns": computed_returns,
                        "truncated_streams": truncated_streams,
                        "count": len(raw_obs),
                    }
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                return None

        # Parallel loading
        print(f"Starting parallel load ({max_workers} workers)...")
        idx = 0
        loaded_files = 0
        computed_returns_files = 0
        truncated_return_files = 0
        truncated_streams_total = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(load_single_file, info): info for info in file_info
            }
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                if result is None:
                    continue

                n = result["count"]
                self.obs[idx : idx + n] = result["obs"]
                self.action[idx : idx + n] = result["action"]
                self.reward[idx : idx + n] = result["reward"]
                self.done[idx : idx + n] = result["done"]
                if self.hidden_state is not None and result["hidden_state"] is not None:
                    self.hidden_state[idx : idx + n] = result["hidden_state"]
                self.return_to_go[idx : idx + n] = result["return_to_go"]
                idx += n
                loaded_files += 1
                computed_returns_files += int(result["computed_returns"])
                if result["truncated_streams"] > 0:
                    truncated_return_files += 1
                    truncated_streams_total += int(result["truncated_streams"])

        self.size = idx
        print(f"Dataset loaded. Total samples: {self.size}")
        print(
            f"Loaded files: {loaded_files}/{len(file_info)} | "
            f"Computed missing returns for {computed_returns_files} files"
        )
        if truncated_return_files > 0:
            print(
                "WARNING: computed returns were truncated at file boundaries for "
                f"{truncated_return_files} files "
                f"({truncated_streams_total} unfinished streams at file end)."
            )
        if self.size == 0:
            raise ValueError("No valid samples were loaded from dataset files.")

        if min_rtg_quantile > 0.0:
            if not (0.0 < min_rtg_quantile < 100.0):
                raise ValueError(
                    f"min_rtg_quantile must be in (0, 100), got {min_rtg_quantile}"
                )
            rtg = self.return_to_go[: self.size]
            threshold = float(np.percentile(rtg, min_rtg_quantile))
            keep_mask = rtg >= threshold
            keep_count = int(np.sum(keep_mask))
            if keep_count == 0:
                raise ValueError(
                    "RTG quantile filter removed all samples; "
                    f"quantile={min_rtg_quantile}, threshold={threshold:.4f}"
                )
            print(
                f"Applied RTG filter at q={min_rtg_quantile:.1f} "
                f"(threshold={threshold:.4f}); keeping {keep_count}/{self.size} samples."
            )
            self.obs = self.obs[: self.size][keep_mask]
            self.action = self.action[: self.size][keep_mask]
            self.reward = self.reward[: self.size][keep_mask]
            self.done = self.done[: self.size][keep_mask]
            if self.hidden_state is not None:
                self.hidden_state = self.hidden_state[: self.size][keep_mask]
            self.return_to_go = self.return_to_go[: self.size][keep_mask]
            self.size = keep_count
        else:
            self.obs = self.obs[: self.size]
            self.action = self.action[: self.size]
            self.reward = self.reward[: self.size]
            self.done = self.done[: self.size]
            if self.hidden_state is not None:
                self.hidden_state = self.hidden_state[: self.size]
            self.return_to_go = self.return_to_go[: self.size]

        self.hidden_refresh_count = None
        if self.hidden_state is not None:
            self.hidden_state, self.hidden_refresh_count = apply_hidden_skip_schedule(
                hidden_state=self.hidden_state,
                done=self.done,
                num_envs=num_envs,
                skip_n=self.hidden_skip_n,
                reset_on_done=self.hidden_skip_reset_on_done,
            )
            total_positions = int(self.hidden_state.shape[0])
            refresh_ratio = (
                float(self.hidden_refresh_count) / float(total_positions)
                if total_positions > 0
                else 0.0
            )
            print(
                "Applied hidden skip schedule: "
                f"refreshes={self.hidden_refresh_count}/{total_positions} "
                f"({refresh_ratio:.4f}), skip_n={self.hidden_skip_n}"
            )

        # Compute normalization statistics for hidden-state inputs.
        if self.hidden_state is not None:
            self.hidden_mean = np.mean(self.hidden_state, axis=0)
            self.hidden_std = np.std(self.hidden_state, axis=0)
            self.hidden_std = np.where(self.hidden_std < 1e-6, 1.0, self.hidden_std)
            print(
                f"Hidden state stats - Mean range: [{self.hidden_mean.min():.3f}, {self.hidden_mean.max():.3f}], "
                f"Std range: [{self.hidden_std.min():.3f}, {self.hidden_std.max():.3f}]"
            )
        else:
            self.hidden_mean = np.zeros((Config.HIDDEN_STATE_DIM,), dtype=np.float32)
            self.hidden_std = np.ones((Config.HIDDEN_STATE_DIM,), dtype=np.float32)
            print("Hidden mode is zero; skipping hidden-state loading and using zero/one normalization stats.")

        # Save normalization statistics for evaluation
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        stats_path = os.path.join(Config.SAVE_DIR, "hidden_state_stats.npz")
        np.savez(stats_path, mean=self.hidden_mean, std=self.hidden_std)
        print(f"Saved normalization statistics to {stats_path}")

    def metadata(self) -> dict:
        file_hash = hashlib.sha1(
            "\n".join(self._dataset_files).encode("utf-8")
        ).hexdigest()
        md = {
            "num_samples": int(self.size),
            "num_files": int(len(self._dataset_files)),
            "dataset_files_sha1": file_hash,
            "hidden_mode": self.hidden_mode,
            "hidden_skip_n": int(self.hidden_skip_n),
            "hidden_skip_reset_on_done": bool(self.hidden_skip_reset_on_done),
        }
        if self.hidden_refresh_count is not None:
            md["hidden_refresh_count"] = int(self.hidden_refresh_count)
            md["hidden_refresh_fraction"] = float(
                self.hidden_refresh_count / max(1, self.size)
            )
        return md

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        obs_t = torch.tensor(self.obs[idx], dtype=torch.float32, device=Config.DEVICE)
        
        action_t = torch.tensor(
            self.action[idx], dtype=torch.long, device=Config.DEVICE
        )

        # Normalize hidden states
        if self.hidden_mode == "real":
            if self.hidden_state is None:
                hidden_normalized = np.zeros(
                    (batch_size, Config.HIDDEN_STATE_DIM), dtype=np.float32
                )
            else:
                hidden_normalized = (
                    self.hidden_state[idx] - self.hidden_mean
                ) / self.hidden_std
        elif self.hidden_mode == "zero":
            hidden_normalized = np.zeros(
                (batch_size, Config.HIDDEN_STATE_DIM), dtype=np.float32
            )
        else:  # shuffle
            if self.hidden_state is None:
                raise ValueError("hidden_mode=shuffle requires hidden states to be loaded.")
            shuffled_idx = np.random.permutation(idx)
            hidden_normalized = (
                self.hidden_state[shuffled_idx] - self.hidden_mean
            ) / self.hidden_std
        hidden_t = torch.tensor(
            hidden_normalized, dtype=torch.float32, device=Config.DEVICE
        )

        rtg_t = torch.tensor(
            self.return_to_go[idx], dtype=torch.float32, device=Config.DEVICE
        )

        return {
            "obs": obs_t,
            "action": action_t,
            "hidden_state": hidden_t,
            "return_to_go": rtg_t,
        }


# ==============================================================================
# 4. Training Step
# ==============================================================================
def train_step(model, optimizer, batch, advantage_mode: str):
    pi, current_v = model(batch["obs"], batch["hidden_state"])

    td_target = batch["return_to_go"]
    advantage = td_target - current_v

    # Critic loss
    critic_loss = 0.5 * torch.mean(advantage.pow(2))

    # Actor loss (AWR)
    log_probs = pi.log_prob(batch["action"])
    detached_advantage = advantage.detach()
    if advantage_mode == "raw":
        weighted_advantage = detached_advantage
    elif advantage_mode == "center":
        weighted_advantage = detached_advantage - detached_advantage.mean()
    elif advantage_mode == "standardize":
        adv_std = detached_advantage.std(unbiased=False).clamp_min(1e-6)
        weighted_advantage = (detached_advantage - detached_advantage.mean()) / adv_std
    else:
        raise ValueError(
            f"Unsupported advantage_mode={advantage_mode}. "
            "Expected one of: raw, center, standardize."
        )

    weights = torch.exp(weighted_advantage / Config.AWR_BETA)
    weights_clipped = torch.clamp(weights, max=Config.AWR_MAX_WEIGHT)
    actor_loss = -torch.mean(log_probs * weights_clipped)

    total_loss = critic_loss + actor_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Metrics
    with torch.no_grad():
        clip_frac = (weights >= Config.AWR_MAX_WEIGHT).float().mean().item()
        var_diff = torch.var(advantage)
        var_return = torch.var(batch["return_to_go"])
        explained_var = 1.0 - (var_diff / (var_return + 1e-8))

        # Hidden state statistics (normalized, as fed to model)
        hidden_mean = batch["hidden_state"].mean().item()
        hidden_std = batch["hidden_state"].std().item()
        hidden_min = batch["hidden_state"].min().item()
        hidden_max = batch["hidden_state"].max().item()
        hidden_gate = None
        if getattr(model, "hidden_gate_logit", None) is not None:
            hidden_gate = torch.sigmoid(model.hidden_gate_logit).item()

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": pi.entropy().mean().item(),
        "mean_weight": weights_clipped.mean().item(),
        "weight_clip_frac": clip_frac,
        "mean_value": current_v.detach().mean().item(),
        "mean_return": batch["return_to_go"].mean().item(),
        "explained_variance": explained_var.item(),
        "hidden_mean": hidden_mean,
        "hidden_std": hidden_std,
        "hidden_min": hidden_min,
        "hidden_max": hidden_max,
        "hidden_gate": hidden_gate,
        "adv_mean": detached_advantage.mean().item(),
        "adv_std": detached_advantage.std(unbiased=False).item(),
    }


# ==============================================================================
# 5. CLI Arguments
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="LLM-Augmented AWR for Craftax")
    parser.add_argument("--data_dir", type=str, default=Config.DATA_DIR)
    parser.add_argument("--data_glob", type=str, default=Config.DATA_GLOB)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--dataset_workers", type=int, default=8)
    parser.add_argument("--num_envs", type=int, default=Config.NUM_ENVS)
    parser.add_argument(
        "--max_dataset_gb",
        type=float,
        default=Config.MAX_DATASET_GB,
        help="Target upper bound for in-memory dataset buffers (GiB).",
    )
    parser.add_argument(
        "--disable_auto_file_limit",
        action="store_true",
        help="If set, fail when estimated dataset memory exceeds --max_dataset_gb.",
    )
    parser.add_argument(
        "--hidden_mode",
        type=str,
        default=Config.HIDDEN_MODE,
        choices=["real", "zero", "shuffle"],
        help="How hidden states are fed during training.",
    )
    parser.add_argument(
        "--hidden_skip_n",
        type=int,
        default=1,
        help="Hold hidden vectors for N timesteps before refreshing (1 disables hold).",
    )
    parser.add_argument(
        "--hidden_skip_reset_on_done",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When true, refresh hidden on first step after episode termination.",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default=Config.FUSION_MODE,
        choices=["concat_raw", "gated_proj", "residual_gated"],
        help="Hidden-state fusion architecture.",
    )
    parser.add_argument(
        "--hidden_gate_init_logit",
        type=float,
        default=Config.HIDDEN_GATE_INIT_LOGIT,
        help="Initial gate logit when --fusion_mode gated_proj.",
    )
    parser.add_argument("--save_dir", type=str, default=Config.SAVE_DIR)
    parser.add_argument("--total_steps", type=int, default=Config.TOTAL_STEPS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--awr_beta", type=float, default=Config.AWR_BETA)
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Custom WandB run name (default: auto-generated with timestamp)",
    )
    parser.add_argument("--save_freq", type=int, default=Config.SAVE_FREQ)
    parser.add_argument(
        "--advantage_mode",
        type=str,
        default=Config.ADVANTAGE_MODE,
        choices=["raw", "center", "standardize"],
        help="Transform applied to advantages before AWR weighting.",
    )
    parser.add_argument(
        "--min_rtg_quantile",
        type=float,
        default=0.0,
        help="Keep only samples with return_to_go >= this percentile threshold (0 disables).",
    )
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--compute-missing-returns",
        dest="compute_missing_returns",
        action="store_true",
        help="If return_to_go is missing in a file, compute it from reward/done.",
    )
    parser.add_argument(
        "--require-returns",
        dest="compute_missing_returns",
        action="store_false",
        help="Fail on files missing return_to_go.",
    )
    parser.set_defaults(compute_missing_returns=True)
    return parser.parse_args()


# ==============================================================================
# 6. Main Training Loop
# ==============================================================================
def main():
    args = parse_args()
    if args.hidden_skip_n < 1:
        raise ValueError(f"--hidden_skip_n must be >= 1, got {args.hidden_skip_n}")

    # Update config
    Config.DATA_DIR = args.data_dir
    Config.DATA_GLOB = args.data_glob
    Config.SAVE_DIR = args.save_dir
    Config.TOTAL_STEPS = args.total_steps
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    Config.AWR_BETA = args.awr_beta
    Config.NUM_ENVS = args.num_envs
    Config.HIDDEN_MODE = args.hidden_mode
    Config.FUSION_MODE = args.fusion_mode
    Config.HIDDEN_GATE_INIT_LOGIT = args.hidden_gate_init_logit
    Config.ADVANTAGE_MODE = args.advantage_mode
    Config.SEED = args.seed
    Config.SAVE_FREQ = args.save_freq

    # Set unique wandb name with timestamp
    if args.wandb_name:
        Config.WANDB_NAME = args.wandb_name
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        Config.WANDB_NAME = f"awr-augmented-llm-{timestamp}"

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    if not args.no_wandb:
        if wandb is None:
            raise ImportError(
                "wandb is not installed in this environment; pass --no_wandb or install wandb."
            )
        wandb.init(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            name=Config.WANDB_NAME,
            config={k: v for k, v in vars(Config).items() if not k.startswith("_")},
            settings=wandb.Settings(init_timeout=300),
        )
        print(f"WandB initialized: {Config.WANDB_PROJECT}/{Config.WANDB_NAME}")

    print(f"Starting LLM-Augmented AWR training with {Config.TOTAL_STEPS} steps")
    print(f"Data: {Config.DATA_DIR}")
    print(f"Data glob: {Config.DATA_GLOB}")
    print(f"Checkpoints: {Config.SAVE_DIR}")
    print(f"AWR Beta: {Config.AWR_BETA}")
    print(f"Hidden mode: {Config.HIDDEN_MODE}")
    print(
        f"Fusion mode: {Config.FUSION_MODE} "
        f"(hidden_gate_init_logit={Config.HIDDEN_GATE_INIT_LOGIT})"
    )
    print(f"Advantage mode: {Config.ADVANTAGE_MODE}")
    print(f"RTG filter quantile: {args.min_rtg_quantile}")
    print(f"Dataset memory budget: {args.max_dataset_gb:.2f} GiB")
    print(
        "Auto file limiting: "
        + ("disabled" if args.disable_auto_file_limit else "enabled")
    )
    if args.max_files is not None:
        print(f"Limiting to first {args.max_files} files")
    
    # Initialize dataset first to determine observation dimension
    print("\n" + "=" * 60)
    print("Loading dataset (this may take several minutes)...")
    print("=" * 60)
    dataset = OfflineDatasetLLMAugmented(
        Config.DATA_DIR,
        Config.DATA_GLOB,
        max_files=args.max_files,
        num_envs=Config.NUM_ENVS,
        compute_missing_returns=args.compute_missing_returns,
        max_workers=args.dataset_workers,
        hidden_mode=args.hidden_mode,
        hidden_skip_n=args.hidden_skip_n,
        hidden_skip_reset_on_done=args.hidden_skip_reset_on_done,
        max_dataset_gb=args.max_dataset_gb,
        auto_file_limit=not args.disable_auto_file_limit,
        min_rtg_quantile=args.min_rtg_quantile,
    )
    print("\n" + "=" * 60)
    print("Dataset loaded successfully!")
    print("=" * 60)

    print("\nInitializing model...")
    model = ActorCriticAug(
        obs_dim=Config.OBS_DIM,
        action_dim=Config.ACTION_DIM,
        layer_width=Config.LAYER_WIDTH,
        hidden_state_dim=Config.HIDDEN_STATE_DIM,
        fusion_mode=Config.FUSION_MODE,
        hidden_gate_init_logit=Config.HIDDEN_GATE_INIT_LOGIT,
    ).to(Config.DEVICE)
    print("Model initialized successfully!")

    print("Creating optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    print("Optimizer created!")

    train_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": vars(args),
        "config": {
            k: v for k, v in vars(Config).items() if not k.startswith("_")
        },
        "dataset": dataset.metadata(),
    }
    meta_path = os.path.join(Config.SAVE_DIR, "training_metadata.json")
    def _json_default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2, sort_keys=True, default=_json_default)
    print(f"Wrote training metadata to {meta_path}")

    model.train()

    for step in range(1, Config.TOTAL_STEPS + 1):
        batch = dataset.sample(Config.BATCH_SIZE)
        metrics = train_step(model, optimizer, batch, Config.ADVANTAGE_MODE)

        if step % Config.LOG_FREQ == 0:
            log_dict = {
                "train/actor_loss": metrics["actor_loss"],
                "train/critic_loss": metrics["critic_loss"],
                "train/mean_weight": metrics["mean_weight"],
                "train/weight_clip_frac": metrics["weight_clip_frac"],
                "train/explained_variance": metrics["explained_variance"],
                "train/adv_mean": metrics["adv_mean"],
                "train/adv_std": metrics["adv_std"],
                "value_debug/predicted_value": metrics["mean_value"],
                "value_debug/actual_return": metrics["mean_return"],
                "hidden_states/mean": metrics["hidden_mean"],
                "hidden_states/std": metrics["hidden_std"],
                "hidden_states/min": metrics["hidden_min"],
                "hidden_states/max": metrics["hidden_max"],
            }
            if not args.no_wandb:
                wandb.log(log_dict, step=step)
                if metrics["hidden_gate"] is not None:
                    wandb.log({"hidden_states/gate": metrics["hidden_gate"]}, step=step)
            if step % (Config.LOG_FREQ * 10) == 0:
                gate_suffix = ""
                if metrics["hidden_gate"] is not None:
                    gate_suffix = f", gate={metrics['hidden_gate']:.4f}"
                print(
                    f"Step {step}/{Config.TOTAL_STEPS}: actor={metrics['actor_loss']:.4f}, "
                    f"critic={metrics['critic_loss']:.4f}, expl_var={metrics['explained_variance']:.3f}, "
                    f"hidden_mean={metrics['hidden_mean']:.4f}, hidden_std={metrics['hidden_std']:.4f}"
                    f"{gate_suffix}"
                )

        if step % Config.SAVE_FREQ == 0:
            ckpt_path = os.path.join(Config.SAVE_DIR, f"awr_llm_checkpoint_{step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint at step {step}")

    final_path = os.path.join(Config.SAVE_DIR, "awr_llm_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
