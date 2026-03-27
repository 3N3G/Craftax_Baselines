"""Unaugmented AWR offline RL — same architecture as PPO ActorCritic (no hidden states).

Architecture: obs→512→512→512→action / obs→512→512→512→value
Dataset: same trajectory .npz files produced by PPO recording runs,
         but hidden_state columns are ignored entirely.
"""

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
import torch.optim as optim
from torch.distributions import Categorical


# ==============================================================================
# 1. Configuration
# ==============================================================================
class Config:
    DATA_DIR = "/data/group_data/rl/geney/vllm_craftax_labelled_results"
    DATA_GLOB = "trajectories_batch_*.npz"

    # Model
    ACTION_DIM = 43
    LAYER_WIDTH = 512
    OBS_DIM = 1345
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
    SAVE_DIR = "/data/group_data/rl/geney/checkpoints/awr_unaugmented/"
    SEED = 42
    MAX_DATASET_GB = 100.0

    # Wandb
    WANDB_PROJECT = "craftax-offline-awr"
    WANDB_ENTITY = "iris-sobolmark"


# ==============================================================================
# Shared helpers (same as awr_llm_augmented.py)
# ==============================================================================
def compute_return_to_go(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    num_envs: int,
) -> Tuple[np.ndarray, int]:
    """Compute return-to-go for flattened trajectories."""
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


# ==============================================================================
# 2. Model Architecture (Unaugmented MLP, matching PPO ActorCritic)
# ==============================================================================
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ActorCriticUnaugmented(nn.Module):
    """Unaugmented actor-critic matching JAX ActorCritic architecture.

    actor: obs→512→512→512→action_dim
    critic: obs→512→512→512→1
    """
    def __init__(self, obs_dim, action_dim, layer_width=512):
        super().__init__()
        self.actor_fc1 = nn.Linear(obs_dim, layer_width)
        self.actor_fc2 = nn.Linear(layer_width, layer_width)
        self.actor_fc3 = nn.Linear(layer_width, layer_width)
        self.actor_out = nn.Linear(layer_width, action_dim)

        self.critic_fc1 = nn.Linear(obs_dim, layer_width)
        self.critic_fc2 = nn.Linear(layer_width, layer_width)
        self.critic_fc3 = nn.Linear(layer_width, layer_width)
        self.critic_out = nn.Linear(layer_width, 1)

        self.apply_init()

    def apply_init(self):
        orthogonal_init(self.actor_fc1, gain=np.sqrt(2))
        orthogonal_init(self.actor_fc2, gain=np.sqrt(2))
        orthogonal_init(self.actor_fc3, gain=np.sqrt(2))
        orthogonal_init(self.actor_out, gain=0.01)

        orthogonal_init(self.critic_fc1, gain=np.sqrt(2))
        orthogonal_init(self.critic_fc2, gain=np.sqrt(2))
        orthogonal_init(self.critic_fc3, gain=np.sqrt(2))
        orthogonal_init(self.critic_out, gain=1.0)

    def forward(self, obs):
        actor_x = torch.tanh(self.actor_fc1(obs))
        actor_x = torch.tanh(self.actor_fc2(actor_x))
        actor_x = torch.tanh(self.actor_fc3(actor_x))
        actor_logits = self.actor_out(actor_x)
        pi = Categorical(logits=actor_logits)

        critic_x = torch.tanh(self.critic_fc1(obs))
        critic_x = torch.tanh(self.critic_fc2(critic_x))
        critic_x = torch.tanh(self.critic_fc3(critic_x))
        value = self.critic_out(critic_x)

        return pi, value.squeeze(-1)


# ==============================================================================
# 3. Dataset Loader (Unaugmented — no hidden states)
# ==============================================================================
class OfflineDatasetUnaugmented:
    def __init__(
        self,
        data_dir,
        file_pattern,
        max_files=None,
        num_envs=128,
        compute_missing_returns=True,
        max_workers=8,
        max_dataset_gb=80.0,
        auto_file_limit=True,
        min_rtg_quantile=0.0,
    ):
        self.num_envs = int(num_envs)
        self.compute_missing_returns = bool(compute_missing_returns)
        self.max_workers = int(max_workers)
        self.max_dataset_gb = max_dataset_gb
        self.auto_file_limit = bool(auto_file_limit)
        self.min_rtg_quantile = float(min_rtg_quantile)

        search_path = os.path.join(data_dir, file_pattern)
        files = glob.glob(search_path)
        if not files:
            raise ValueError(f"No files found at {search_path}")

        files = sorted(files)
        if max_files is not None:
            files = files[:max_files]
        print(f"Found {len(files)} trajectory files.")

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

        # Count total samples.
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

        self._all_file_info = list(file_info)
        self._all_dataset_files = [f for f, _ in self._all_file_info]
        # Compute bytes per sample AFTER obs_dim is known from the first file.
        # Config.OBS_DIM was updated above by the first-file scan.
        self._bytes_per_sample = (
            4 * int(Config.OBS_DIM)  # obs (float32)
            + 3 * 4                  # action(int32), reward(f32), done(f32)
            + 4                      # return_to_go (float32)
        )
        print(f"Bytes per sample: {self._bytes_per_sample} (obs_dim={Config.OBS_DIM})")
        estimated_gb = self._estimate_buffer_gb(total_samples)
        print(f"Estimated in-memory dataset footprint: {estimated_gb:.2f} GiB")

        if (
            self.max_dataset_gb is not None
            and estimated_gb > self.max_dataset_gb
            and not self.auto_file_limit
        ):
            raise MemoryError(
                "Estimated dataset buffers exceed memory budget. "
                f"Need ~{estimated_gb:.2f} GiB for {total_samples} samples, "
                f"budget is {self.max_dataset_gb:.2f} GiB. "
                "Increase --max_dataset_gb or keep auto file limiting enabled."
            )

        self._shards = self._build_file_shards(self._all_file_info)
        self._current_shard_idx = 0
        self._dataset_files = []

        if len(self._shards) == 1:
            shard_samples = sum(n for _, n in self._shards[0])
            print(
                "Dataset fits in one shard: "
                f"{len(self._shards[0])} files, {shard_samples} samples, "
                f"~{self._estimate_buffer_gb(shard_samples):.2f} GiB."
            )
        else:
            max_shard_samples = max(sum(n for _, n in shard) for shard in self._shards)
            print(
                "Dataset sharded for bounded memory: "
                f"{len(self._shards)} shards over {len(self._all_file_info)} files; "
                f"max shard ~{self._estimate_buffer_gb(max_shard_samples):.2f} GiB "
                f"(budget {self.max_dataset_gb:.2f} GiB)."
            )

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None
        self.return_to_go = None
        self.size = 0

        self._load_shard(self._current_shard_idx)

    def _estimate_buffer_gb(self, sample_count: int) -> float:
        return (sample_count * self._bytes_per_sample) / (1024 ** 3)

    def _build_file_shards(self, file_info):
        if self.max_dataset_gb is None:
            return [list(file_info)]

        shards = []
        current = []
        current_samples = 0
        for info in file_info:
            n = int(info[1])
            if n <= 0:
                continue
            single_gb = self._estimate_buffer_gb(n)
            if single_gb > self.max_dataset_gb:
                raise MemoryError(
                    "Dataset exceeds memory budget and even one file does not fit. "
                    f"File={info[0]}, needs ~{single_gb:.2f} GiB, "
                    f"budget={self.max_dataset_gb:.2f} GiB."
                )

            next_samples = current_samples + n
            if current and self._estimate_buffer_gb(next_samples) > self.max_dataset_gb:
                shards.append(current)
                current = [info]
                current_samples = n
            else:
                current.append(info)
                current_samples = next_samples

        if current:
            shards.append(current)
        if not shards:
            raise ValueError("No shardable files found after metadata scan.")
        return shards

    @property
    def num_shards(self) -> int:
        return len(self._shards)

    @property
    def current_shard_index(self) -> int:
        return int(self._current_shard_idx)

    def _load_single_file(self, args):
        fpath, _expected_n = args
        try:
            with np.load(fpath) as data:
                raw_obs = data["obs"]
                if len(raw_obs.shape) > 2:
                    raw_obs = raw_obs.reshape(raw_obs.shape[0], -1)

                if "return_to_go" in data:
                    return_to_go = data["return_to_go"].astype(np.float32)
                    computed_returns = False
                    truncated_streams = 0
                elif self.compute_missing_returns and "reward" in data and "done" in data:
                    return_to_go, truncated_streams = compute_return_to_go(
                        data["reward"], data["done"], Config.GAMMA, self.num_envs
                    )
                    computed_returns = True
                else:
                    raise KeyError(
                        "Missing return_to_go and cannot compute it "
                        "(require reward/done or enable --compute-missing-returns)."
                    )

                return {
                    "path": fpath,
                    "obs": raw_obs.astype(np.float32),
                    "action": data["action"],
                    "reward": data["reward"].astype(np.float32),
                    "done": data["done"],
                    "return_to_go": return_to_go,
                    "computed_returns": computed_returns,
                    "truncated_streams": truncated_streams,
                    "count": len(raw_obs),
                }
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            return None

    def _load_shard(self, shard_idx: int):
        shard_info = self._shards[shard_idx]
        shard_samples = int(sum(n for _, n in shard_info))
        print(
            f"Loading shard {shard_idx + 1}/{self.num_shards}: "
            f"{len(shard_info)} files, {shard_samples} samples "
            f"(~{self._estimate_buffer_gb(shard_samples):.2f} GiB)"
        )
        print(f"Allocating buffers for {shard_samples} samples...")

        self.obs = np.zeros((shard_samples, Config.OBS_DIM), dtype=np.float32)
        self.action = np.zeros((shard_samples,), dtype=np.int32)
        self.reward = np.zeros((shard_samples,), dtype=np.float32)
        self.done = np.zeros((shard_samples,), dtype=np.float32)
        self.return_to_go = np.zeros((shard_samples,), dtype=np.float32)

        print(f"Starting parallel load ({self.max_workers} workers)...")
        idx = 0
        loaded_files = 0
        computed_returns_files = 0
        truncated_return_files = 0
        truncated_streams_total = 0
        loaded_file_paths = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._load_single_file, info): info for info in shard_info
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
                self.return_to_go[idx : idx + n] = result["return_to_go"]
                idx += n
                loaded_files += 1
                loaded_file_paths.append(result["path"])
                computed_returns_files += int(result["computed_returns"])
                if result["truncated_streams"] > 0:
                    truncated_return_files += 1
                    truncated_streams_total += int(result["truncated_streams"])

        self.size = idx
        print(f"Dataset shard loaded. Total samples: {self.size}")
        print(
            f"Loaded files: {loaded_files}/{len(shard_info)} | "
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

        if self.min_rtg_quantile > 0.0:
            if not (0.0 < self.min_rtg_quantile < 100.0):
                raise ValueError(
                    f"min_rtg_quantile must be in (0, 100), got {self.min_rtg_quantile}"
                )
            rtg = self.return_to_go[: self.size]
            threshold = float(np.percentile(rtg, self.min_rtg_quantile))
            keep_mask = rtg >= threshold
            keep_count = int(np.sum(keep_mask))
            if keep_count == 0:
                raise ValueError(
                    "RTG quantile filter removed all samples; "
                    f"quantile={self.min_rtg_quantile}, threshold={threshold:.4f}"
                )
            print(
                f"Applied RTG filter at q={self.min_rtg_quantile:.1f} "
                f"(threshold={threshold:.4f}); keeping {keep_count}/{self.size} samples."
            )
            self.obs = self.obs[: self.size][keep_mask]
            self.action = self.action[: self.size][keep_mask]
            self.reward = self.reward[: self.size][keep_mask]
            self.done = self.done[: self.size][keep_mask]
            self.return_to_go = self.return_to_go[: self.size][keep_mask]
            self.size = keep_count
        else:
            self.obs = self.obs[: self.size]
            self.action = self.action[: self.size]
            self.reward = self.reward[: self.size]
            self.done = self.done[: self.size]
            self.return_to_go = self.return_to_go[: self.size]

        loaded_file_paths = sorted(loaded_file_paths)
        self._current_shard_idx = int(shard_idx)
        self._dataset_files = loaded_file_paths

    def advance_shard(self) -> bool:
        if self.num_shards <= 1:
            return False
        next_idx = (self._current_shard_idx + 1) % self.num_shards
        self._load_shard(next_idx)
        return True

    def metadata(self) -> dict:
        all_hash = hashlib.sha1(
            "\n".join(self._all_dataset_files).encode("utf-8")
        ).hexdigest()
        shard_hash = hashlib.sha1(
            "\n".join(self._dataset_files).encode("utf-8")
        ).hexdigest()
        return {
            "num_samples": int(self.size),
            "num_files": int(len(self._dataset_files)),
            "num_files_total": int(len(self._all_dataset_files)),
            "num_shards": int(self.num_shards),
            "current_shard_index": int(self.current_shard_index),
            "dataset_files_sha1": all_hash,
            "current_shard_files_sha1": shard_hash,
        }

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        obs_t = torch.tensor(self.obs[idx], dtype=torch.float32, device=Config.DEVICE)
        action_t = torch.tensor(
            self.action[idx], dtype=torch.long, device=Config.DEVICE
        )
        rtg_t = torch.tensor(
            self.return_to_go[idx], dtype=torch.float32, device=Config.DEVICE
        )

        return {
            "obs": obs_t,
            "action": action_t,
            "return_to_go": rtg_t,
        }


# ==============================================================================
# 4. Training Step
# ==============================================================================
def train_step(model, optimizer, batch, advantage_mode: str):
    pi, current_v = model(batch["obs"])

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

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": pi.entropy().mean().item(),
        "mean_weight": weights_clipped.mean().item(),
        "weight_clip_frac": clip_frac,
        "mean_value": current_v.detach().mean().item(),
        "mean_return": batch["return_to_go"].mean().item(),
        "explained_variance": explained_var.item(),
        "adv_mean": detached_advantage.mean().item(),
        "adv_std": detached_advantage.std(unbiased=False).item(),
    }


# ==============================================================================
# 5. CLI Arguments
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Unaugmented AWR for Craftax (no hidden states)")
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
        "--dataset_shard_steps",
        type=int,
        default=0,
        help="Steps to train before rotating to next shard (<=0 => auto).",
    )
    parser.add_argument(
        "--disable_dataset_shard_rotation",
        action="store_true",
        help="If set, keep training on the initial shard only.",
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

    # Update config
    Config.DATA_DIR = args.data_dir
    Config.DATA_GLOB = args.data_glob
    Config.SAVE_DIR = args.save_dir
    Config.TOTAL_STEPS = args.total_steps
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    Config.AWR_BETA = args.awr_beta
    Config.NUM_ENVS = args.num_envs
    Config.ADVANTAGE_MODE = args.advantage_mode
    Config.SEED = args.seed
    Config.SAVE_FREQ = args.save_freq

    # Set unique wandb name with timestamp
    if args.wandb_name:
        Config.WANDB_NAME = args.wandb_name
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        Config.WANDB_NAME = f"awr-unaugmented-{timestamp}"

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

    print(f"Starting Unaugmented AWR training with {Config.TOTAL_STEPS} steps")
    print(f"Data: {Config.DATA_DIR}")
    print(f"Data glob: {Config.DATA_GLOB}")
    print(f"Checkpoints: {Config.SAVE_DIR}")
    print(f"AWR Beta: {Config.AWR_BETA}")
    print("Architecture: ActorCriticUnaugmented (obs→512→512→512→action/value)")
    print(f"Advantage mode: {Config.ADVANTAGE_MODE}")
    print(f"RTG filter quantile: {args.min_rtg_quantile}")
    print(f"Dataset memory budget: {args.max_dataset_gb:.2f} GiB")
    print(
        "Auto file limiting: "
        + ("disabled" if args.disable_auto_file_limit else "enabled")
    )
    if args.max_files is not None:
        print(f"Limiting to first {args.max_files} files")

    # Initialize dataset
    print("\n" + "=" * 60)
    print("Loading dataset (this may take several minutes)...")
    print("=" * 60)
    dataset = OfflineDatasetUnaugmented(
        Config.DATA_DIR,
        Config.DATA_GLOB,
        max_files=args.max_files,
        num_envs=Config.NUM_ENVS,
        compute_missing_returns=args.compute_missing_returns,
        max_workers=args.dataset_workers,
        max_dataset_gb=args.max_dataset_gb,
        auto_file_limit=not args.disable_auto_file_limit,
        min_rtg_quantile=args.min_rtg_quantile,
    )
    print("\n" + "=" * 60)
    print("Dataset loaded successfully!")
    print("=" * 60)

    dataset_shard_steps = None
    if dataset.num_shards > 1 and not args.disable_dataset_shard_rotation:
        if args.dataset_shard_steps > 0:
            dataset_shard_steps = int(args.dataset_shard_steps)
        else:
            dataset_shard_steps = int(np.ceil(Config.TOTAL_STEPS / dataset.num_shards))
            dataset_shard_steps = max(1, dataset_shard_steps)
        print(
            "Dataset shard rotation: enabled "
            f"({dataset.num_shards} shards, rotate every {dataset_shard_steps} steps)."
        )
    elif dataset.num_shards > 1:
        print(
            "Dataset shard rotation: disabled by flag; "
            f"training will stay on shard 1/{dataset.num_shards}."
        )
    else:
        print("Dataset shard rotation: single-shard dataset (no rotation needed).")

    print("\nInitializing model...")
    model = ActorCriticUnaugmented(
        obs_dim=Config.OBS_DIM,
        action_dim=Config.ACTION_DIM,
        layer_width=Config.LAYER_WIDTH,
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
        "dataset_rotation": {
            "enabled": bool(dataset_shard_steps is not None),
            "dataset_shard_steps": int(dataset_shard_steps) if dataset_shard_steps is not None else None,
            "num_shards": int(dataset.num_shards),
        },
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
        if dataset_shard_steps is not None and step > 1 and (step - 1) % dataset_shard_steps == 0:
            t_reload = time.time()
            rotated = dataset.advance_shard()
            reload_sec = time.time() - t_reload
            if rotated:
                print(
                    f"Rotated dataset to shard {dataset.current_shard_index + 1}/{dataset.num_shards} "
                    f"at step {step} (reload {reload_sec:.1f}s)."
                )
                if not args.no_wandb:
                    wandb.log(
                        {
                            "dataset/current_shard": float(dataset.current_shard_index + 1),
                            "dataset/num_shards": float(dataset.num_shards),
                            "dataset/shard_samples": float(dataset.size),
                            "dataset/reload_seconds": float(reload_sec),
                        },
                        step=step,
                    )

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
                "dataset/current_shard": float(dataset.current_shard_index + 1),
                "dataset/num_shards": float(dataset.num_shards),
                "dataset/shard_samples": float(dataset.size),
            }
            if not args.no_wandb:
                wandb.log(log_dict, step=step)
            if step % (Config.LOG_FREQ * 10) == 0:
                print(
                    f"Step {step}/{Config.TOTAL_STEPS}: actor={metrics['actor_loss']:.4f}, "
                    f"critic={metrics['critic_loss']:.4f}, expl_var={metrics['explained_variance']:.3f}"
                )

        if step % Config.SAVE_FREQ == 0:
            ckpt_path = os.path.join(Config.SAVE_DIR, f"awr_unaugmented_checkpoint_{step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint at step {step}")

    final_path = os.path.join(Config.SAVE_DIR, "awr_unaugmented_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
