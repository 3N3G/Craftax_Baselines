#!/usr/bin/env python3
"""
Evaluate offline symbolic LLM-augmented torch policies in Craftax.

Supports hidden input modes:
- zero: feed all-zero hidden vectors
- llm:  feed live hidden states from a local vLLM server
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch
import requests

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from craftax.craftax_env import make_craftax_env_from_name  # noqa: E402
from offline_rl.awr_llm_augmented import ActorCriticAug  # noqa: E402
from labelling.obs_to_text import obs_to_text  # noqa: E402
from utils.llm_extractor import VLLMHiddenStateExtractor  # noqa: E402
from utils.llm_prompts import filter_text_obs  # noqa: E402
from utils.wrappers import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper  # noqa: E402


class LLMHiddenStateManagerLite:
    def __init__(self, model_id: str, target_layer: int, tokens_to_generate: int):
        self.tokens_to_generate = tokens_to_generate
        vllm_url = "http://localhost:8000"
        try:
            resp = requests.get(f"{vllm_url}/health", timeout=2)
            if resp.status_code != 200:
                raise RuntimeError(f"vLLM health returned {resp.status_code}")
        except Exception as e:
            raise RuntimeError(
                f"vLLM server unavailable at {vllm_url}. "
                "Start with: bash scripts/start_vllm_hidden.sh --mode last_token"
            ) from e

        model_name = "./configs/vllm_hidden_qwen4b"
        extracted_layers = [8, 16, 24, 35]
        layer_index = -1
        if target_layer != -1 and target_layer in extracted_layers:
            layer_index = extracted_layers.index(target_layer)

        self.llm = VLLMHiddenStateExtractor(
            server_url=vllm_url,
            model_name=model_name,
            model_id=model_id,
            target_layer=layer_index,
        )
        self.hidden_size = self.llm.hidden_size

    def extract(self, obs_batch: jnp.ndarray, num_envs: int) -> Tuple[jnp.ndarray, Dict]:
        obs_host = np.asarray(jax.device_get(obs_batch))
        prompts = []
        for i in range(num_envs):
            raw_text = obs_to_text(obs_host[i])
            prompts.append(filter_text_obs(raw_text))

        if self.tokens_to_generate == 1:
            hidden_np, metrics = self.llm.extract_hidden_states_no_cot(prompts)
        else:
            hidden_np, _, metrics = self.llm.extract_hidden_states(
                prompts,
                batch_size=min(32, len(prompts)),
                max_new_tokens=self.tokens_to_generate,
            )
        return jnp.asarray(hidden_np), metrics


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def flatten(prefix: str, data: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in data.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(flatten(key, v))
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[key] = float(v)
    return out


def extract_done_episode_achievements(info: Dict, done_mask: np.ndarray) -> Tuple[List[int], Dict[str, float]]:
    ach_keys = sorted(k for k in info.keys() if k.startswith("Achievements/"))
    if not ach_keys:
        return [], {}

    ach_cols = []
    for k in ach_keys:
        ach_cols.append(np.asarray(jax.device_get(info[k]), dtype=np.float32))
    ach_mat = np.stack(ach_cols, axis=-1)  # [num_envs, num_achievements]

    done_idx = np.nonzero(done_mask)[0]
    episode_counts: List[int] = []
    unlock_sum = np.zeros((ach_mat.shape[1],), dtype=np.float64)
    for i in done_idx:
        unlocked = ach_mat[i] > 0.0
        episode_counts.append(int(unlocked.sum()))
        unlock_sum += unlocked.astype(np.float64)

    rates = {}
    denom = max(1, len(done_idx))
    for j, k in enumerate(ach_keys):
        rates[k] = float(unlock_sum[j] / denom)
    return episode_counts, rates


def infer_fusion_mode(state_dict: Dict[str, torch.Tensor], hidden_dim: int) -> str:
    if "actor_obs_fc1.weight" in state_dict and "actor_hidden_fc1.weight" in state_dict:
        return "dual_concat"

    actor_in = int(state_dict["actor_fc1.weight"].shape[1])
    layer_width = int(state_dict["encoder_fc1.weight"].shape[0])
    if actor_in == layer_width + hidden_dim:
        return "concat_raw"
    if actor_in == 2 * layer_width:
        return "gated_proj"
    if actor_in == layer_width:
        return "residual_gated"
    raise ValueError(
        f"Could not infer fusion mode from actor_fc1.in_features={actor_in}, "
        f"layer_width={layer_width}, hidden_dim={hidden_dim}"
    )


def load_policy(
    checkpoint_path: Path,
    stats_path: Path | None,
    device: torch.device,
) -> Tuple[ActorCriticAug, np.ndarray, np.ndarray, Dict]:
    state_dict = torch.load(checkpoint_path, map_location=device)

    if "encoder_fc1.weight" in state_dict:
        obs_dim = int(state_dict["encoder_fc1.weight"].shape[1])
        layer_width = int(state_dict["encoder_fc1.weight"].shape[0])
    elif "actor_obs_fc1.weight" in state_dict:
        obs_dim = int(state_dict["actor_obs_fc1.weight"].shape[1])
        layer_width = int(state_dict["actor_obs_fc1.weight"].shape[0])
    else:
        raise KeyError("Unable to infer obs/layer dims from checkpoint state_dict")

    if "actor_out.weight" in state_dict:
        action_dim = int(state_dict["actor_out.weight"].shape[0])
    elif "actor_fc2.weight" in state_dict:
        action_dim = int(state_dict["actor_fc2.weight"].shape[0])
    else:
        raise KeyError("Unable to infer action_dim from checkpoint state_dict")

    inferred_hidden_dim = None
    if "hidden_proj.weight" in state_dict:
        inferred_hidden_dim = int(state_dict["hidden_proj.weight"].shape[1])
    elif "actor_hidden_fc1.weight" in state_dict:
        inferred_hidden_dim = int(state_dict["actor_hidden_fc1.weight"].shape[1])

    if stats_path is not None and stats_path.exists():
        stats = np.load(stats_path)
        hidden_mean = np.asarray(stats["mean"], dtype=np.float32)
        hidden_std = np.asarray(stats["std"], dtype=np.float32)
        hidden_dim = int(hidden_mean.shape[0])
    elif inferred_hidden_dim is not None:
        hidden_dim = inferred_hidden_dim
        hidden_mean = np.zeros((hidden_dim,), dtype=np.float32)
        hidden_std = np.ones((hidden_dim,), dtype=np.float32)
    else:
        raise ValueError(
            "Unable to determine hidden dim. Provide stats file or checkpoint with hidden_proj.weight."
        )

    model = ActorCriticAug(
        obs_dim=obs_dim,
        action_dim=action_dim,
        layer_width=layer_width,
        hidden_state_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    hidden_std = np.where(hidden_std < 1e-6, 1.0, hidden_std).astype(np.float32)
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "stats_path": str(stats_path) if stats_path is not None else None,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "layer_width": layer_width,
        "hidden_dim": hidden_dim,
        "fusion_mode": "fixed_dual_branch",
    }
    return model, hidden_mean.astype(np.float32), hidden_std, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate offline LLM-augmented symbolic policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--stats-path",
        type=str,
        default=None,
        help="Path to hidden_state_stats.npz (default: checkpoint_dir/hidden_state_stats.npz).",
    )
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-env-steps", type=int, default=20000)
    parser.add_argument(
        "--hidden-input-mode",
        type=str,
        default="zero",
        choices=["zero", "llm"],
        help="How hidden states are supplied during evaluation.",
    )
    parser.add_argument("--skip-n", type=int, default=1, help="LLM refresh period in steps when hidden-input-mode=llm.")
    parser.add_argument("--llm-layer", type=int, default=-1)
    parser.add_argument("--llm-tokens", type=int, default=1)
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax action instead of sampling.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="craftax_symbolic_evals")
    parser.add_argument("--wandb-entity", type=str, default="iris-sobolmark")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.stats_path is None:
        stats_path = ckpt_path.parent / "hidden_state_stats.npz"
        if not stats_path.exists():
            stats_path = None
    else:
        stats_path = Path(args.stats_path).expanduser().resolve()

    device = torch.device(args.device)
    model, hidden_mean, hidden_std, model_meta = load_policy(
        checkpoint_path=ckpt_path, stats_path=stats_path, device=device
    )
    hidden_dim = int(hidden_mean.shape[0])

    llm_manager = None
    if args.hidden_input_mode == "llm":
        llm_manager = LLMHiddenStateManagerLite(
            model_id=args.llm_model,
            target_layer=args.llm_layer,
            tokens_to_generate=args.llm_tokens,
        )
        if llm_manager.hidden_size != hidden_dim:
            raise ValueError(
                f"Hidden dim mismatch: policy expects {hidden_dim}, "
                f"LLM extractor produced {llm_manager.hidden_size}."
            )

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=args.num_envs)

    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)

    hidden_raw = np.zeros((args.num_envs, hidden_dim), dtype=np.float32)
    hidden_norm = np.zeros((args.num_envs, hidden_dim), dtype=np.float32)
    steps_since_llm = args.skip_n

    episode_returns: List[float] = []
    episode_lengths: List[float] = []
    episode_achievements: List[int] = []
    achievement_sum: Dict[str, float] = {}
    achievement_count: Dict[str, int] = {}

    env_steps = 0
    while len(episode_returns) < args.num_episodes and env_steps < args.max_env_steps:
        obs_np = np.asarray(jax.device_get(obs), dtype=np.float32).copy()

        if llm_manager is not None and steps_since_llm >= args.skip_n:
            hidden_jnp, _ = llm_manager.extract(obs, args.num_envs)
            hidden_raw = np.asarray(jax.device_get(hidden_jnp), dtype=np.float32)
            steps_since_llm = 0
        hidden_norm = (hidden_raw - hidden_mean[None, :]) / hidden_std[None, :]
        if llm_manager is not None:
            steps_since_llm += 1

        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
            hidden_t = torch.from_numpy(hidden_norm).to(device=device, dtype=torch.float32)
            pi, _ = model(obs_t, hidden_t)
            if args.deterministic:
                actions_t = torch.argmax(pi.logits, dim=-1)
            else:
                actions_t = pi.sample()
        actions = actions_t.cpu().numpy().astype(np.int32)

        rng, _rng = jax.random.split(rng)
        obs, env_state, _, _, info = env.step(_rng, env_state, jnp.asarray(actions), env_params)
        env_steps += 1
        if env_steps % 25 == 0:
            print(
                f"[eval] env_steps={env_steps} completed_episodes={len(episode_returns)}/{args.num_episodes}",
                flush=True,
            )

        done_mask = np.asarray(jax.device_get(info["returned_episode"]), dtype=bool)
        if np.any(done_mask):
            rets = np.asarray(jax.device_get(info["returned_episode_returns"]), dtype=np.float32)
            lens = np.asarray(jax.device_get(info["returned_episode_lengths"]), dtype=np.float32)
            done_idx = np.nonzero(done_mask)[0]
            for i in done_idx:
                episode_returns.append(float(rets[i]))
                episode_lengths.append(float(lens[i]))
                if len(episode_returns) >= args.num_episodes:
                    break

            per_ep_ach_counts, per_step_ach_rates = extract_done_episode_achievements(info, done_mask)
            for c in per_ep_ach_counts:
                episode_achievements.append(int(c))
            for k, v in per_step_ach_rates.items():
                achievement_sum[k] = achievement_sum.get(k, 0.0) + float(v)
                achievement_count[k] = achievement_count.get(k, 0) + 1

    if len(episode_returns) == 0:
        raise RuntimeError("No completed episodes during evaluation window.")

    episode_returns = episode_returns[: args.num_episodes]
    episode_lengths = episode_lengths[: args.num_episodes]
    episode_achievements = episode_achievements[: args.num_episodes]

    achievement_rates = {}
    for k, s in achievement_sum.items():
        n = achievement_count.get(k, 0)
        if n > 0:
            achievement_rates[k] = float(s / n)

    results = {
        "config": {
            "num_episodes": args.num_episodes,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "max_env_steps": args.max_env_steps,
            "hidden_input_mode": args.hidden_input_mode,
            "skip_n": args.skip_n,
            "deterministic": bool(args.deterministic),
            "device": str(device),
        },
        "model": model_meta,
        "completed_episodes": int(len(episode_returns)),
        "env_steps": int(env_steps),
        "returns": summarize(episode_returns),
        "episode_lengths": summarize(episode_lengths),
        "achievements_per_episode": summarize(episode_achievements),
        "achievement_unlock_rates": achievement_rates,
    }

    print(json.dumps(results, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"Wrote eval summary to {output_path}")

    if not args.no_wandb:
        if wandb is None:
            raise ImportError("wandb not installed; rerun with --no-wandb or install wandb.")
        run_name = args.wandb_name or f"offline_eval_{ckpt_path.stem}_{args.hidden_input_mode}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "checkpoint": str(ckpt_path),
                "stats_path": None if stats_path is None else str(stats_path),
                **results["config"],
                **results["model"],
            },
        )
        wandb.log(
            {
                "eval/completed_episodes": float(results["completed_episodes"]),
                "eval/env_steps": float(results["env_steps"]),
                **flatten("eval/returns", results["returns"]),
                **flatten("eval/episode_lengths", results["episode_lengths"]),
                **flatten("eval/achievements_per_episode", results["achievements_per_episode"]),
                **{
                    f"eval/achievement_unlock_rate/{k.replace('/', '_')}": float(v)
                    for k, v in results["achievement_unlock_rates"].items()
                },
            }
        )
        wandb.summary.update(results)
        wandb.finish()


if __name__ == "__main__":
    main()
