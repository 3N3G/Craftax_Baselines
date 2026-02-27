#!/usr/bin/env python3
"""
Evaluate symbolic Craftax policies with consistent 128-episode metrics and W&B logging.

Policies evaluated by default:
  1) online-jax skip25 (LLM hidden states)
  2) online-jax skip100000000 (zero hidden baseline)
  3) PPO Orbax checkpoint
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Keep JAX memory usage predictable when running alongside vLLM.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
try:
    import torch
except ImportError:
    torch = None
import wandb
import yaml
from flax import serialization
from flax.training.train_state import TrainState

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from craftax.craftax.constants import BLOCK_PIXEL_SIZE_AGENT  # noqa: E402
from craftax.craftax.renderer import render_craftax_pixels  # noqa: E402
from craftax.craftax_env import make_craftax_env_from_name  # noqa: E402
from labelling.obs_to_text import obs_to_text  # noqa: E402
from models.actor_critic import ActorCritic, ActorCriticAug  # noqa: E402
from online_rl_llm.online_rl_hidden_jax import LLMHiddenStateManager  # noqa: E402
from utils.llm_prompts import filter_text_obs  # noqa: E402

TorchActorCriticAug = None  # lazy import; loaded on demand by _load_torch_offline_policy
from utils.wrappers import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper  # noqa: E402

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


@dataclass
class PolicySpec:
    name: str
    policy_type: str  # "aug_msgpack" | "torch_offline_aug" | "torch_offline_unaugmented" | "ppo_orbax" | "ppo_msgpack"
    policy_path: str
    metadata_path: Optional[str]
    skip_n: int
    hidden_mode: str  # "llm" | "zero" | "none"
    train_step: Optional[int] = None
    stats_path: Optional[str] = None
    llm_layer: int = -1


def _latest_existing(paths: List[str]) -> Optional[str]:
    uniq = sorted({p for p in paths if p})
    if not uniq:
        return None
    uniq.sort(key=lambda p: (os.path.getmtime(p), p))
    # Prefer finalized checkpoints over periodic snapshots when both exist.
    non_step = [p for p in uniq if "_step" not in Path(p).name]
    return non_step[-1] if non_step else uniq[-1]


def _discover_online_ckpt(tokens: List[str], suffix: str) -> Optional[str]:
    roots = [
        "/data/group_data/rl/geney/online_rl_hidden_models",
        "/data/group_data/rl/geney/online_rl_hidden_models/short_ablation",
    ]
    candidates: List[str] = []
    for root in roots:
        for token in tokens:
            pattern = f"{root}/**/*{token}*{suffix}"
            candidates.extend(glob.glob(pattern, recursive=True))
    return _latest_existing(candidates)


def _resolve_online_paths(
    ckpt_path: str,
    meta_path: Optional[str],
    tokens: List[str],
) -> Tuple[str, Optional[str]]:
    resolved_ckpt = ckpt_path
    if not Path(resolved_ckpt).exists():
        found = _discover_online_ckpt(tokens=tokens, suffix=".msgpack")
        if found is None:
            raise FileNotFoundError(
                f"Could not resolve checkpoint for tokens={tokens}. "
                f"Tried explicit path: {ckpt_path}"
            )
        resolved_ckpt = found

    resolved_meta = meta_path
    if resolved_meta is None:
        candidate = str(Path(resolved_ckpt).with_suffix(".json"))
        if Path(candidate).exists():
            resolved_meta = candidate
    elif not Path(resolved_meta).exists():
        candidate = str(Path(resolved_ckpt).with_suffix(".json"))
        if Path(candidate).exists():
            resolved_meta = candidate
        else:
            found_meta = _discover_online_ckpt(tokens=tokens, suffix=".json")
            resolved_meta = found_meta if found_meta is not None else None

    return resolved_ckpt, resolved_meta


def _unwrap_wandb_cfg(raw: Dict) -> Dict:
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
        else:
            out[k] = v
    return out


def _summarize(values: List[float]) -> Dict[str, float]:
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


def _flatten(prefix: str, data: Dict) -> Dict[str, float]:
    flat = {}
    for k, v in data.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            flat.update(_flatten(key, v))
        elif isinstance(v, (int, float, np.integer, np.floating)):
            flat[key] = float(v)
    return flat


def _load_online_training_metadata(path: Optional[str]) -> Dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {"metadata_missing": path}
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "source": str(p),
        "final_return_logged": payload.get("final_return"),
        "sps_logged": payload.get("sps"),
        "llm_calls_logged": payload.get("llm_calls"),
        "metadata": payload.get("metadata", {}),
    }


def _load_ppo_training_metadata(run_dir: str, step: int) -> Dict:
    run_path = Path(run_dir)
    cfg_path = run_path / "config.yaml"
    out = {
        "run_dir": str(run_path),
        "checkpoint_step": int(step),
    }
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
        cfg = _unwrap_wandb_cfg(raw_cfg)
        out["config"] = cfg
    else:
        out["config_missing"] = str(cfg_path)
    return out


def _load_offline_training_metadata(checkpoint_path: str, stats_path: Optional[str], llm_layer: int) -> Dict:
    out = {
        "checkpoint": str(checkpoint_path),
        "llm_layer": int(llm_layer),
    }
    if stats_path:
        out["stats_path"] = str(stats_path)
        if not Path(stats_path).exists():
            out["stats_missing"] = True
    return out


def _training_scalar_summary(training_summary: Dict) -> Dict[str, float]:
    scalars: Dict[str, float] = {}
    for key in ("final_return_logged", "sps_logged", "llm_calls_logged"):
        if isinstance(training_summary.get(key), (int, float)):
            scalars[f"model/{key}"] = float(training_summary[key])

    metadata = training_summary.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("timesteps", "envs", "skip_n", "num_steps", "tokens", "llm_calls"):
            val = metadata.get(key)
            if isinstance(val, (int, float)):
                scalars[f"model/metadata_{key}"] = float(val)

    cfg = training_summary.get("config", {})
    if isinstance(cfg, dict):
        for key in ("TOTAL_TIMESTEPS", "NUM_ENVS", "NUM_STEPS", "UPDATE_EPOCHS", "NUM_MINIBATCHES"):
            val = cfg.get(key)
            if isinstance(val, (int, float)):
                scalars[f"model/config_{key.lower()}"] = float(val)

    return scalars


def _load_aug_policy(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
    layer_width: int,
):
    ckpt = Path(checkpoint_path)
    with ckpt.open("rb") as f:
        blob = f.read()

    net = ActorCriticAug(
        action_dim=action_dim,
        layer_width=layer_width,
        hidden_state_dim=hidden_dim,
    )
    template = net.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        jnp.zeros((1, hidden_dim), dtype=jnp.float32),
    )
    params = serialization.from_bytes(template, blob)

    @jax.jit
    def act_fn(params, obs, hidden, rng):
        pi, _ = net.apply(params, obs, hidden)
        return pi.sample(seed=rng)

    @jax.jit
    def value_fn(params, obs, hidden):
        _, value = net.apply(params, obs, hidden)
        return value

    return {
        "params": params,
        "act_fn": act_fn,
        "value_fn": value_fn,
        "fusion_mode_loaded": "fixed_dual_branch",
        "uses_hidden": True,
        "hidden_mean": np.zeros((hidden_dim,), dtype=np.float32),
        "hidden_std": np.ones((hidden_dim,), dtype=np.float32),
    }


def _load_torch_offline_policy(checkpoint_path: str, stats_path: Optional[str]):
    global TorchActorCriticAug
    if torch is None:
        raise ImportError("torch is required for torch_offline_aug policy type")
    if TorchActorCriticAug is None:
        from offline_rl.awr_llm_augmented import ActorCriticAug as TorchActorCriticAug
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    else:
        state_dict = raw
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unexpected checkpoint format for {checkpoint_path}")

    if "actor_obs_fc1.weight" not in state_dict:
        raise KeyError("Missing actor_obs_fc1.weight in offline checkpoint")
    layer_width = int(state_dict["actor_obs_fc1.weight"].shape[0])
    obs_dim = int(state_dict["actor_obs_fc1.weight"].shape[1])
    if "actor_out.weight" in state_dict:
        action_dim = int(state_dict["actor_out.weight"].shape[0])
    elif "actor_fc2.weight" in state_dict:
        action_dim = int(state_dict["actor_fc2.weight"].shape[0])
    else:
        raise KeyError("Unable to infer action_dim from checkpoint state_dict")

    if "hidden_proj.weight" in state_dict:
        hidden_dim = int(state_dict["hidden_proj.weight"].shape[1])
    elif "actor_hidden_fc1.weight" in state_dict:
        hidden_dim = int(state_dict["actor_hidden_fc1.weight"].shape[1])
    else:
        raise KeyError("Unable to infer hidden_dim from checkpoint state_dict")

    hidden_mean = np.zeros((hidden_dim,), dtype=np.float32)
    hidden_std = np.ones((hidden_dim,), dtype=np.float32)
    if stats_path and Path(stats_path).exists():
        stats = np.load(stats_path)
        hidden_mean = np.asarray(stats["mean"], dtype=np.float32)
        hidden_std = np.asarray(stats["std"], dtype=np.float32)
    hidden_std = np.where(hidden_std < 1e-6, 1.0, hidden_std)

    model = TorchActorCriticAug(
        obs_dim=obs_dim,
        action_dim=action_dim,
        layer_width=layer_width,
        hidden_state_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    def act_fn(params, obs, hidden, rng):
        del params, rng
        obs_np = np.asarray(jax.device_get(obs), dtype=np.float32)
        hid_np = np.asarray(jax.device_get(hidden), dtype=np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
            hid_t = torch.from_numpy(hid_np).to(device=device, dtype=torch.float32)
            pi, _ = model(obs_t, hid_t)
            actions = pi.sample().detach().cpu().numpy().astype(np.int32)
        return jnp.asarray(actions, dtype=jnp.int32)

    def value_fn(params, obs, hidden):
        del params
        obs_np = np.asarray(jax.device_get(obs), dtype=np.float32)
        hid_np = np.asarray(jax.device_get(hidden), dtype=np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
            hid_t = torch.from_numpy(hid_np).to(device=device, dtype=torch.float32)
            _, value = model(obs_t, hid_t)
            value_np = value.detach().cpu().numpy().astype(np.float32)
        return jnp.asarray(value_np, dtype=jnp.float32)

    return {
        "params": None,
        "act_fn": act_fn,
        "value_fn": value_fn,
        "uses_hidden": True,
        "hidden_mean": hidden_mean,
        "hidden_std": hidden_std,
        "framework": "torch",
    }


def _load_ppo_policy(run_dir: str, step: int, obs_dim: int, action_dim: int):
    run_path = Path(run_dir)
    cfg_path = run_path / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    cfg = _unwrap_wandb_cfg(raw_cfg)

    layer_size = int(cfg.get("LAYER_SIZE", 512))
    lr = float(cfg.get("LR", 2e-4))
    max_grad_norm = float(cfg.get("MAX_GRAD_NORM", 1.0))

    net = ActorCritic(action_dim=action_dim, layer_width=layer_size)
    params0 = net.init(jax.random.PRNGKey(0), jnp.zeros((1, obs_dim), dtype=jnp.float32))
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr, eps=1e-5))
    train_state = TrainState.create(apply_fn=net.apply, params=params0, tx=tx)

    ckpt_mgr = ocp.CheckpointManager(
        str(run_path / "policies"),
        ocp.PyTreeCheckpointer(),
        ocp.CheckpointManagerOptions(max_to_keep=1, create=False),
    )
    train_state = ckpt_mgr.restore(step, items=train_state)

    @jax.jit
    def act_fn(params, obs, rng):
        pi, _ = net.apply(params, obs)
        return pi.sample(seed=rng)

    @jax.jit
    def value_fn(params, obs):
        _, value = net.apply(params, obs)
        return value

    return {
        "params": train_state.params,
        "act_fn": act_fn,
        "value_fn": value_fn,
        "layer_size": layer_size,
        "uses_hidden": False,
        "hidden_mean": None,
        "hidden_std": None,
    }


def _load_ppo_msgpack_policy(checkpoint_path: str, obs_dim: int, action_dim: int, layer_width: int = 512):
    ckpt = Path(checkpoint_path)
    blob = ckpt.read_bytes()

    net = ActorCritic(action_dim=action_dim, layer_width=layer_width)
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((1, obs_dim), dtype=jnp.float32))
    params = serialization.from_bytes(template, blob)

    @jax.jit
    def act_fn(params, obs, rng):
        pi, _ = net.apply(params, obs)
        return pi.sample(seed=rng)

    @jax.jit
    def value_fn(params, obs):
        _, value = net.apply(params, obs)
        return value

    return {
        "params": params,
        "act_fn": act_fn,
        "value_fn": value_fn,
        "uses_hidden": False,
        "hidden_mean": None,
        "hidden_std": None,
    }


def _load_torch_offline_unaugmented_policy(checkpoint_path: str):
    if torch is None:
        raise ImportError("torch is required for torch_offline_unaugmented policy type")
    from offline_rl.awr_unaugmented import ActorCriticUnaugmented

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    else:
        state_dict = raw
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unexpected checkpoint format for {checkpoint_path}")

    # Infer dimensions from state_dict
    obs_dim = int(state_dict["actor_fc1.weight"].shape[1])
    layer_width = int(state_dict["actor_fc1.weight"].shape[0])
    action_dim = int(state_dict["actor_out.weight"].shape[0])

    model = ActorCriticUnaugmented(
        obs_dim=obs_dim,
        action_dim=action_dim,
        layer_width=layer_width,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    def act_fn(params, obs, rng):
        del params, rng
        obs_np = np.asarray(jax.device_get(obs), dtype=np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
            pi, _ = model(obs_t)
            actions = pi.sample().detach().cpu().numpy().astype(np.int32)
        return jnp.asarray(actions, dtype=jnp.int32)

    def value_fn(params, obs):
        del params
        obs_np = np.asarray(jax.device_get(obs), dtype=np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
            _, value = model(obs_t)
            value_np = value.detach().cpu().numpy().astype(np.float32)
        return jnp.asarray(value_np, dtype=jnp.float32)

    return {
        "params": None,
        "act_fn": act_fn,
        "value_fn": value_fn,
        "uses_hidden": False,
        "hidden_mean": None,
        "hidden_std": None,
        "framework": "torch",
    }


def _overlay_text(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    if cv2 is None:
        return frame
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 18
    for line in lines:
        cv2.putText(out, line, (8, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16
    return out


def _fit_text_to_width(text: str, max_width: int, font_scale: float, thickness: int) -> str:
    if cv2 is None:
        return text
    clean = " ".join(str(text).split())
    if not clean:
        return ""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, _), _ = cv2.getTextSize(clean, font, font_scale, thickness)
    if w <= max_width:
        return clean

    suffix = "..."
    lo, hi = 0, len(clean)
    best = suffix
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = clean[:mid].rstrip() + suffix
        (cw, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if cw <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _compute_empirical_rtg(rewards: List[float], gamma: float) -> np.ndarray:
    rtg = np.zeros((len(rewards),), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = float(rewards[i]) + gamma * running
        rtg[i] = running
    return rtg


def _normalize_hidden(
    hidden: jnp.ndarray,
    hidden_mean: Optional[np.ndarray],
    hidden_std: Optional[np.ndarray],
) -> jnp.ndarray:
    if hidden_mean is None or hidden_std is None:
        return hidden
    mean = jnp.asarray(hidden_mean, dtype=jnp.float32)
    std = jnp.asarray(hidden_std, dtype=jnp.float32)
    return (hidden - mean[None, :]) / std[None, :]


def _sanitize_llm_text(text: str, max_words: int) -> str:
    s = " ".join(str(text).replace("\n", " ").split())
    if not s:
        return "LLM: (no generated text)"
    s = s.replace("<think>", "").replace("</think>", "").strip()
    if not s:
        return "LLM: (no generated text)"
    # Keep the first sentence-like span and hard-cap words so the full
    # LLM string can fit above the game without visual truncation.
    s = re.split(r"[.!?]", s, maxsplit=1)[0].strip()
    words = s.split()
    if max_words > 0 and len(words) > max_words:
        s = " ".join(words[:max_words]).strip()
    if not s:
        return "LLM: (no generated text)"
    return f"LLM: {s}"


def _generate_llm_text_for_video(
    llm_manager: LLMHiddenStateManager, obs_single: jnp.ndarray, max_new_tokens: int
) -> str:
    obs_np = np.asarray(jax.device_get(obs_single), dtype=np.float32)
    prompt = filter_text_obs(obs_to_text(obs_np))

    if max_new_tokens > 1:
        try:
            _, generated, _ = llm_manager.llm.extract_hidden_states(
                [prompt],
                batch_size=1,
                max_new_tokens=max_new_tokens,
            )
            if generated:
                return str(generated[0])
        except Exception:
            pass

    return ""


def _render_video_frame(
    game_frame: np.ndarray,
    status_line: str,
    llm_line: str,
    value_hist: List[float],
    rtg_hist: List[float],
    total_steps: int,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    if cv2 is None:
        return game_frame

    frame = np.ascontiguousarray(game_frame)
    gh, gw, _ = frame.shape

    header_h = 40
    footer_h = 88
    canvas = np.zeros((header_h + gh + footer_h, gw, 3), dtype=np.uint8)

    # Header above game grid.
    canvas[0:header_h, :, :] = (18, 18, 18)
    canvas[header_h : header_h + gh, :, :] = frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    status = _fit_text_to_width(status_line, gw - 12, font_scale=0.40, thickness=1)
    llm = " ".join(str(llm_line).split())
    cv2.putText(canvas, status, (6, 15), font, 0.40, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, llm, (6, 33), font, 0.31, (180, 220, 255), 1, cv2.LINE_AA)

    graph_top = header_h + gh + 8
    graph_bottom = header_h + gh + footer_h - 10
    graph_left = 10
    graph_right = gw - 10
    cv2.rectangle(canvas, (graph_left, graph_top), (graph_right, graph_bottom), (35, 35, 35), -1)
    cv2.rectangle(canvas, (graph_left, graph_top), (graph_right, graph_bottom), (70, 70, 70), 1)

    if y_max <= y_min + 1e-6:
        y_max = y_min + 1e-6

    def to_xy(idx: int, value: float) -> Tuple[int, int]:
        denom = max(1, total_steps - 1)
        x = int(graph_left + (idx / denom) * (graph_right - graph_left))
        ratio = float((value - y_min) / (y_max - y_min))
        ratio = min(1.0, max(0.0, ratio))
        y = int(graph_bottom - ratio * (graph_bottom - graph_top))
        return x, y

    for i in range(1, len(value_hist)):
        x1, y1 = to_xy(i - 1, float(value_hist[i - 1]))
        x2, y2 = to_xy(i, float(value_hist[i]))
        cv2.line(canvas, (x1, y1), (x2, y2), (60, 220, 60), 2, cv2.LINE_AA)

    for i in range(1, len(rtg_hist)):
        x1, y1 = to_xy(i - 1, float(rtg_hist[i - 1]))
        x2, y2 = to_xy(i, float(rtg_hist[i]))
        cv2.line(canvas, (x1, y1), (x2, y2), (70, 140, 255), 2, cv2.LINE_AA)

    v_disp = float(value_hist[-1]) if value_hist else 0.0
    r_disp = float(rtg_hist[-1]) if rtg_hist else 0.0
    cv2.putText(
        canvas,
        f"V(s): {v_disp:.2f}  RTG: {r_disp:.2f}  [green=value, blue=empirical RTG]",
        (graph_left + 4, graph_top + 14),
        font,
        0.36,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _extract_done_episode_achievements(info: Dict, done_mask: np.ndarray) -> Tuple[List[int], Dict[str, float]]:
    ach_keys = sorted(k for k in info.keys() if k.startswith("Achievements/"))
    if not ach_keys:
        return [], {}
    ach_cols = []
    for k in ach_keys:
        ach_cols.append(np.asarray(jax.device_get(info[k]), dtype=np.float32))
    ach_mat = np.stack(ach_cols, axis=-1)  # [num_envs, num_ach]
    done_idx = np.nonzero(done_mask)[0]
    episode_counts = []
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


def evaluate_policy(
    spec: PolicySpec,
    args,
    llm_manager: Optional[LLMHiddenStateManager],
    obs_dim: int,
    action_dim: int,
):
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=args.num_envs)

    if spec.policy_type == "aug_msgpack":
        policy = _load_aug_policy(
            checkpoint_path=spec.policy_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            layer_width=args.layer_width,
        )
    elif spec.policy_type == "torch_offline_aug":
        policy = _load_torch_offline_policy(
            checkpoint_path=spec.policy_path,
            stats_path=spec.stats_path,
        )
    elif spec.policy_type == "ppo_orbax":
        policy = _load_ppo_policy(
            run_dir=spec.policy_path,
            step=int(spec.train_step),
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    elif spec.policy_type == "ppo_msgpack":
        policy = _load_ppo_msgpack_policy(
            checkpoint_path=spec.policy_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            layer_width=args.layer_width,
        )
    elif spec.policy_type == "torch_offline_unaugmented":
        policy = _load_torch_offline_unaugmented_policy(
            checkpoint_path=spec.policy_path,
        )
    else:
        raise ValueError(f"Unsupported policy type: {spec.policy_type}")

    uses_hidden = bool(policy.get("uses_hidden", False))
    hidden_mean = policy.get("hidden_mean")
    hidden_std = policy.get("hidden_std")
    if uses_hidden and hidden_mean is not None:
        hidden_dim = int(np.asarray(hidden_mean).shape[0])
    else:
        hidden_dim = int(args.hidden_dim)

    rng = jax.random.PRNGKey(args.seed)
    rng, rr = jax.random.split(rng)
    obs, env_state = env.reset(rr, env_params)

    hidden = jnp.zeros((args.num_envs, hidden_dim), dtype=jnp.float32)
    steps_since_llm = spec.skip_n
    llm_calls = 0
    llm_time_ms = []
    text_time_ms = []

    returns = []
    lengths = []
    ach_counts = []
    ach_rate_totals: Dict[str, float] = {}
    episode_counter = 0

    start = time.time()
    steps = 0
    while episode_counter < args.target_episodes and steps < args.max_env_steps:
        if uses_hidden:
            if spec.hidden_mode == "llm":
                if llm_manager is None:
                    raise RuntimeError("LLM manager required for hidden_mode=llm")
                if steps_since_llm >= spec.skip_n:
                    hidden_raw, hm = llm_manager.extract(obs, args.num_envs)
                    hidden = _normalize_hidden(hidden_raw, hidden_mean, hidden_std)
                    llm_calls += 1
                    steps_since_llm = 0
                    llm_time_ms.append(float(hm.get("timing/llm_inference_ms", 0.0)))
                    text_time_ms.append(float(hm.get("timing/text_render_ms", 0.0)))
            elif spec.hidden_mode == "zero":
                hidden = jnp.zeros_like(hidden)
            else:
                raise ValueError(f"Unsupported hidden_mode for hidden policy: {spec.hidden_mode}")

            rng, ar = jax.random.split(rng)
            action = policy["act_fn"](policy["params"], obs, hidden, ar)
            steps_since_llm += 1
        else:
            rng, ar = jax.random.split(rng)
            action = policy["act_fn"](policy["params"], obs, ar)

        rng, sr = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(sr, env_state, action, env_params)
        steps += 1

        done_mask = np.asarray(jax.device_get(done)).astype(bool)
        if done_mask.any():
            ret_np = np.asarray(jax.device_get(info["returned_episode_returns"]), dtype=np.float32)
            len_np = np.asarray(jax.device_get(info["returned_episode_lengths"]), dtype=np.int32)
            done_idx = np.nonzero(done_mask)[0]

            ep_ach_counts, ep_ach_rates = _extract_done_episode_achievements(info, done_mask)
            for k, v in ep_ach_rates.items():
                ach_rate_totals[k] = ach_rate_totals.get(k, 0.0) + v * len(done_idx)

            for local_pos, i in enumerate(done_idx):
                returns.append(float(ret_np[i]))
                lengths.append(int(len_np[i]))
                if local_pos < len(ep_ach_counts):
                    ach_counts.append(int(ep_ach_counts[local_pos]))
                else:
                    ach_counts.append(0)
                episode_counter += 1
                if episode_counter >= args.target_episodes:
                    break

        if steps % 200 == 0:
            print(
                f"[{spec.name}] steps={steps} episodes={episode_counter}/{args.target_episodes} llm_calls={llm_calls}",
                flush=True,
            )

    elapsed = time.time() - start

    # Aggregate achievement unlock rates over collected episodes.
    ach_rates = {}
    denom = max(1, episode_counter)
    for k, accum in ach_rate_totals.items():
        ach_rates[k.replace("Achievements/", "").lower()] = float(accum / denom)

    sorted_ach = sorted(ach_rates.items(), key=lambda kv: kv[1], reverse=True)
    top_ach = sorted_ach[:12]

    metrics = {
        "policy_name": spec.name,
        "policy_type": spec.policy_type,
        "policy_path": spec.policy_path,
        "episodes": int(episode_counter),
        "envs": int(args.num_envs),
        "runtime_sec": float(elapsed),
        "env_steps": int(steps),
        "return": _summarize(returns),
        "length": _summarize(lengths),
        "achievement_count": _summarize(ach_counts),
        "achievement_any_unlocked_mean": float(np.mean(np.asarray(ach_counts) > 0.0)) if ach_counts else 0.0,
        "achievement_total_unique_unlocked": int(sum(1 for _, r in ach_rates.items() if r > 0.0)),
        "achievement_unlock_rate": ach_rates,
        "achievement_unlock_rate_top12": [(k, float(v)) for k, v in top_ach],
        "llm_calls": int(llm_calls),
        "llm_ms_mean": float(np.mean(llm_time_ms)) if llm_time_ms else 0.0,
        "text_ms_mean": float(np.mean(text_time_ms)) if text_time_ms else 0.0,
        "returns_raw": returns,
        "lengths_raw": lengths,
        "achievement_count_raw": ach_counts,
    }
    return metrics, policy


def rollout_video(spec: PolicySpec, policy: Dict, args, llm_manager: Optional[LLMHiddenStateManager]):
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params

    rng = jax.random.PRNGKey(args.seed + 999)
    rng, rr = jax.random.split(rng)
    obs, state = env.reset(rr, env_params)

    uses_hidden = bool(policy.get("uses_hidden", False))
    hidden_mean = policy.get("hidden_mean")
    hidden_std = policy.get("hidden_std")
    if uses_hidden and hidden_mean is not None:
        hidden_dim = int(np.asarray(hidden_mean).shape[0])
    else:
        hidden_dim = int(args.hidden_dim)
    hidden = jnp.zeros((1, hidden_dim), dtype=jnp.float32)
    steps_since_llm = spec.skip_n
    llm_calls = 0
    ep_return = 0.0
    raw_frames: List[np.ndarray] = []
    rewards: List[float] = []
    predicted_values: List[float] = []
    ach_counts: List[int] = []
    llm_calls_hist: List[int] = []
    llm_text_hist: List[str] = []
    latest_llm_text = "LLM: (no generated text)"

    for step in range(args.video_max_steps):
        if uses_hidden:
            if spec.hidden_mode == "llm":
                if llm_manager is None:
                    raise RuntimeError("LLM manager required for hidden_mode=llm")
                if steps_since_llm >= spec.skip_n:
                    # Keep policy conditioning identical to training/eval:
                    # hidden comes from the same manager.extract() prefill path.
                    hidden_raw, _ = llm_manager.extract(obs[None, ...], 1)
                    hidden = _normalize_hidden(hidden_raw, hidden_mean, hidden_std)
                    generated_text = _generate_llm_text_for_video(
                        llm_manager=llm_manager,
                        obs_single=obs,
                        max_new_tokens=args.video_llm_tokens,
                    )
                    latest_llm_text = _sanitize_llm_text(generated_text, max_words=args.video_llm_max_words)
                    llm_calls += 1
                    steps_since_llm = 0
            elif spec.hidden_mode == "zero":
                hidden = jnp.zeros_like(hidden)

        if uses_hidden:
            value = float(
                np.asarray(
                    jax.device_get(policy["value_fn"](policy["params"], obs[None, ...], hidden)),
                    dtype=np.float32,
                )[0]
            )
        else:
            value = float(
                np.asarray(
                    jax.device_get(policy["value_fn"](policy["params"], obs[None, ...])),
                    dtype=np.float32,
                )[0]
            )

        frame = np.asarray(
            render_craftax_pixels(state, BLOCK_PIXEL_SIZE_AGENT, do_night_noise=False),
            dtype=np.uint8,
        )
        ach_count = int(np.asarray(jax.device_get(state.achievements)).sum())
        raw_frames.append(frame)
        ach_counts.append(ach_count)
        llm_calls_hist.append(llm_calls)
        llm_text_hist.append(latest_llm_text)
        predicted_values.append(value)

        rng, ar = jax.random.split(rng)
        if uses_hidden:
            action = policy["act_fn"](policy["params"], obs[None, ...], hidden, ar)[0]
            steps_since_llm += 1
        else:
            action = policy["act_fn"](policy["params"], obs[None, ...], ar)[0]

        rng, sr = jax.random.split(rng)
        obs, state, reward, done, info = env.step(sr, state, action, env_params)
        ep_return += float(reward)
        rewards.append(float(reward))

        if bool(jax.device_get(done)):
            break

    if not raw_frames:
        return None

    rtg = _compute_empirical_rtg(rewards, gamma=args.video_gamma)

    m = min(len(raw_frames), len(predicted_values), len(rtg))
    if m == 0:
        return None
    raw_frames = raw_frames[:m]
    predicted_values = predicted_values[:m]
    rtg = rtg[:m]
    ach_counts = ach_counts[:m]
    llm_calls_hist = llm_calls_hist[:m]
    llm_text_hist = llm_text_hist[:m]
    rewards = rewards[:m]

    all_vals = np.concatenate(
        [np.asarray(predicted_values, dtype=np.float32), np.asarray(rtg, dtype=np.float32)]
    )
    y_min = float(np.percentile(all_vals, 2))
    y_max = float(np.percentile(all_vals, 98))
    if y_max <= y_min + 1e-6:
        y_min -= 1.0
        y_max += 1.0

    frames: List[np.ndarray] = []
    total_steps = len(raw_frames)
    cumulative_return = 0.0
    for i in range(total_steps):
        cumulative_return += float(rewards[i])
        status = (
            f"{spec.name} | step={i} | return={cumulative_return:.2f} | "
            f"achievements={ach_counts[i]} | llm_calls={llm_calls_hist[i]}"
        )
        frame = _render_video_frame(
            game_frame=raw_frames[i],
            status_line=status,
            llm_line=llm_text_hist[i],
            value_hist=predicted_values[: i + 1],
            rtg_hist=rtg[: i + 1].tolist(),
            total_steps=total_steps,
            y_min=y_min,
            y_max=y_max,
        )
        frames.append(frame)

    video = np.stack(frames, axis=0)  # [T, H, W, C]
    video = np.transpose(video, (0, 3, 1, 2))  # [T, C, H, W]
    return video


def _get_llm_manager(
    llm_manager_by_layer: Dict[int, LLMHiddenStateManager], args, llm_layer: int
) -> LLMHiddenStateManager:
    layer = int(llm_layer)
    if layer not in llm_manager_by_layer:
        print(f"Initializing shared LLM hidden-state manager for layer={layer}...", flush=True)
        llm_manager_by_layer[layer] = LLMHiddenStateManager(
            model_id="Qwen/Qwen3-4B",
            target_layer=layer,
            tokens_to_generate=args.llm_tokens,
        )
    return llm_manager_by_layer[layer]


def policy_specs_from_manifest(manifest_path: str, manifest_policy_ids: str) -> List[PolicySpec]:
    from scripts.eval_policy_wave import resolve_manifest_policies

    manifest_p = Path(manifest_path).expanduser().resolve()
    manifest = yaml.safe_load(manifest_p.read_text())
    resolved = resolve_manifest_policies(manifest, include_slices=False, slice_count=0)

    policy_cfg = {str(p.get("id")): p for p in manifest.get("policies", [])}
    wanted = [x.strip() for x in str(manifest_policy_ids).split(",") if x.strip()]
    if wanted:
        wanted_set = set(wanted)
        resolved = [r for r in resolved if r.policy_id in wanted_set]
        missing = sorted(wanted_set - {r.policy_id for r in resolved})
        if missing:
            raise ValueError(f"Missing manifest_policy_ids in resolved manifest: {missing}")

    out: List[PolicySpec] = []
    for r in resolved:
        if r.policy_type == "jax_aug_msgpack":
            kind = "aug_msgpack"
        elif r.policy_type == "torch_offline_aug":
            kind = "torch_offline_aug"
        elif r.policy_type == "ppo_orbax":
            kind = "ppo_orbax"
        elif r.policy_type == "ppo_msgpack":
            kind = "ppo_msgpack"
        elif r.policy_type == "torch_offline_unaugmented":
            kind = "torch_offline_unaugmented"
        else:
            continue

        cfg = policy_cfg.get(r.policy_id, {})
        out.append(
            PolicySpec(
                name=str(r.policy_id),
                policy_type=kind,
                policy_path=str(r.checkpoint_path),
                metadata_path=r.metadata_path,
                skip_n=int(r.skip_n),
                hidden_mode=str(r.hidden_mode),
                train_step=r.train_step,
                stats_path=r.stats_path,
                llm_layer=int(cfg.get("llm_layer", -1)),
            )
        )

    if not out:
        raise ValueError(f"No supported policies resolved from manifest: {manifest_p}")
    return out


def default_policy_specs(args) -> List[PolicySpec]:
    skip25_ckpt, skip25_meta = _resolve_online_paths(
        ckpt_path=args.skip25_path,
        meta_path=args.skip25_meta,
        tokens=["online-jax-128env-skip25", "skip25"],
    )
    skip100_ckpt, skip100_meta = _resolve_online_paths(
        ckpt_path=args.skip100m_path,
        meta_path=args.skip100m_meta,
        tokens=["online-jax-128env-skip100000000", "skip100000000", "skip100m"],
    )

    return [
        PolicySpec(
            name="skip25_real_hidden",
            policy_type="aug_msgpack",
            policy_path=skip25_ckpt,
            metadata_path=skip25_meta,
            skip_n=args.skip25_n,
            hidden_mode="llm",
        ),
        PolicySpec(
            name="skip100m_baseline",
            policy_type="aug_msgpack",
            policy_path=skip100_ckpt,
            metadata_path=skip100_meta,
            skip_n=args.skip100m_n,
            hidden_mode="zero",
        ),
        PolicySpec(
            name="ppo_symbolic",
            policy_type="ppo_orbax",
            policy_path=args.ppo_run_dir,
            metadata_path=None,
            skip_n=0,
            hidden_mode="none",
            train_step=args.ppo_step,
        ),
    ]


def select_policy_specs(all_specs: List[PolicySpec], requested_csv: str) -> List[PolicySpec]:
    requested = [x.strip() for x in requested_csv.split(",") if x.strip()]
    if not requested:
        raise ValueError("No policies selected. Pass at least one policy name in --policies.")
    spec_by_name = {s.name: s for s in all_specs}
    unknown = [name for name in requested if name not in spec_by_name]
    if unknown:
        raise ValueError(
            f"Unknown policy names in --policies: {unknown}. "
            f"Available: {sorted(spec_by_name.keys())}"
        )
    return [spec_by_name[name] for name in requested]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Craftax symbolic policies with W&B logging.")
    parser.add_argument("--wandb-project", type=str, default="craftax_symbolic_evals")
    parser.add_argument("--wandb-entity", type=str, default="iris-sobolmark")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--target-episodes", type=int, default=128)
    parser.add_argument("--max-env-steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--layer-width", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument(
        "--llm-tokens",
        type=int,
        default=1,
        help="Hidden-state extraction mode for policy conditioning (1 = no_cot/prefill path).",
    )
    parser.add_argument("--video-max-steps", type=int, default=600)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument(
        "--video-llm-tokens",
        type=int,
        default=1,
        help="Generated tokens for LLM text shown in video HUD (llm-mode policies).",
    )
    parser.add_argument(
        "--video-llm-max-words",
        type=int,
        default=12,
        help="Hard cap for displayed LLM words in video HUD.",
    )
    parser.add_argument(
        "--video-gamma",
        type=float,
        default=0.99,
        help="Discount for empirical RTG overlay curve in videos.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional manifest yaml. If set, policy specs are resolved from manifest.",
    )
    parser.add_argument(
        "--manifest-policy-ids",
        type=str,
        default="",
        help="Optional comma-separated policy ids to include from --manifest.",
    )
    parser.add_argument("--output-json", type=str, default="logs/policy_eval_craftax_symbolic_evals.json")
    parser.add_argument(
        "--policies",
        type=str,
        default="skip25_real_hidden,skip100m_baseline,ppo_symbolic",
        help="Comma-separated subset of policy names to evaluate.",
    )

    parser.add_argument(
        "--skip25-path",
        type=str,
        default="/data/group_data/rl/geney/online_rl_hidden_models/online-jax-128env-skip25_20260222_142300.msgpack",
    )
    parser.add_argument(
        "--skip25-meta",
        type=str,
        default="/data/group_data/rl/geney/online_rl_hidden_models/online-jax-128env-skip25_20260222_142300.json",
    )
    parser.add_argument("--skip25-n", type=int, default=25)

    parser.add_argument(
        "--skip100m-path",
        type=str,
        default="/data/group_data/rl/geney/online_rl_hidden_models/online-jax-128env-skip100000000_20260221_153626.msgpack",
    )
    parser.add_argument(
        "--skip100m-meta",
        type=str,
        default="/data/group_data/rl/geney/online_rl_hidden_models/online-jax-128env-skip100000000_20260221_153626.json",
    )
    parser.add_argument("--skip100m-n", type=int, default=100000000)

    parser.add_argument(
        "--ppo-run-dir",
        type=str,
        default="/home/geney/Craftax_Baselines/wandb/run-20260127_213945-lvd7jvbd/files",
    )
    parser.add_argument("--ppo-step", type=int, default=100000000)
    parser.add_argument("--no-video", action="store_true")

    # Adhoc single policy evaluation
    parser.add_argument("--adhoc-policy-name", type=str, default="")
    parser.add_argument("--adhoc-policy-type", type=str, default="")
    parser.add_argument("--adhoc-policy-path", type=str, default="")
    parser.add_argument("--adhoc-policy-meta", type=str, default="")
    parser.add_argument("--adhoc-policy-skip-n", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(Path(args.output_json).parent, exist_ok=True)

    # Resolve env dimensions once.
    probe_env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = probe_env.default_params
    obs_dim = int(probe_env.observation_space(env_params).shape[0])
    action_dim = int(probe_env.action_space(env_params).n)

    if args.adhoc_policy_name and args.adhoc_policy_type and args.adhoc_policy_path:
        specs = [
            PolicySpec(
                name=args.adhoc_policy_name,
                policy_type=args.adhoc_policy_type,
                policy_path=args.adhoc_policy_path,
                metadata_path=args.adhoc_policy_meta if args.adhoc_policy_meta else None,
                skip_n=args.adhoc_policy_skip_n,
                hidden_mode="raw",  # Default
                llm_layer=25,
            )
        ]
    elif args.manifest:
        specs = policy_specs_from_manifest(
            manifest_path=args.manifest,
            manifest_policy_ids=args.manifest_policy_ids,
        )
    else:
        all_specs = default_policy_specs(args)
        specs = select_policy_specs(all_specs, args.policies)
    llm_manager_by_layer: Dict[int, LLMHiddenStateManager] = {}

    out = {
        "timestamp": datetime.now().isoformat(),
        "seed": int(args.seed),
        "num_envs": int(args.num_envs),
        "target_episodes": int(args.target_episodes),
        "policies": {},
    }

    for spec in specs:
        print(f"\n=== Evaluating {spec.name} ===", flush=True)
        llm_manager = _get_llm_manager(llm_manager_by_layer, args, spec.llm_layer) if spec.hidden_mode == "llm" else None
        if spec.policy_type == "aug_msgpack":
            training_summary = _load_online_training_metadata(spec.metadata_path)
        elif spec.policy_type == "torch_offline_aug":
            training_summary = _load_offline_training_metadata(
                checkpoint_path=spec.policy_path,
                stats_path=spec.stats_path,
                llm_layer=spec.llm_layer,
            )
        elif spec.policy_type == "ppo_orbax":
            training_summary = _load_ppo_training_metadata(spec.policy_path, int(spec.train_step))
        else:
            training_summary = {"checkpoint": spec.policy_path}

        metrics, policy = evaluate_policy(
            spec=spec,
            args=args,
            llm_manager=llm_manager,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        run_name = f"{spec.name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_cfg = {
            "policy_name": spec.name,
            "policy_type": spec.policy_type,
            "policy_path": spec.policy_path,
            "hidden_mode_eval": spec.hidden_mode,
            "skip_n": int(spec.skip_n),
            "llm_layer": int(spec.llm_layer),
            "target_episodes": int(args.target_episodes),
            "num_envs": int(args.num_envs),
            "max_env_steps": int(args.max_env_steps),
            "llm_tokens": int(args.llm_tokens),
            "video_max_steps": int(args.video_max_steps),
            "video_llm_tokens": int(args.video_llm_tokens),
            "video_llm_max_words": int(args.video_llm_max_words),
            "video_gamma": float(args.video_gamma),
            "seed": int(args.seed),
            "training_summary": training_summary,
        }
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=run_cfg,
            reinit=True,
            settings=wandb.Settings(init_timeout=300),
        )
        wandb.log(_flatten("eval", metrics))
        wandb.log(_training_scalar_summary(training_summary))

        if not args.no_video:
            video = rollout_video(spec=spec, policy=policy, args=args, llm_manager=llm_manager)
            if video is not None:
                wandb.log({"eval/video": wandb.Video(video, fps=args.video_fps, format="mp4")})

        # Explicit summary block requested by user.
        run.summary["model/training_summary_json"] = json.dumps(training_summary, sort_keys=True)
        run.summary["eval/final_results_json"] = json.dumps(metrics, sort_keys=True)
        run.finish()

        out["policies"][spec.name] = {
            "training_summary": training_summary,
            "evaluation": metrics,
        }
        print(
            f"{spec.name}: mean_return={metrics['return'].get('mean')} "
            f"mean_ach={metrics['achievement_count'].get('mean')} "
            f"episodes={metrics['episodes']}",
            flush=True,
        )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote evaluation summary: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
