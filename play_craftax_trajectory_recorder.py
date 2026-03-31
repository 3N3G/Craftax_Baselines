import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import jax
import numpy as np
import pygame

from craftax.craftax.constants import Action, BLOCK_PIXEL_SIZE_HUMAN
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.play_craftax import (
    CraftaxRenderer,
    KEY_MAPPING,
    print_new_achievements,
    save_compressed_pickle,
)
from craftax.craftax.renderer import render_craftax_text


def _get_repo_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _action_name(action_id):
    if action_id is None:
        return None
    try:
        return Action(int(action_id)).name
    except Exception:
        return None


class TrajectoryRecorder:
    """Record timestep-wise trajectories with optional full env-state snapshots."""

    def __init__(
        self,
        base_dir: str = "play_data/trajectory_records",
        env_name: str = "Craftax-Symbolic-v1",
        env_params_digest: str = "unknown",
        repo_commit: str = "unknown",
    ):
        self.schema_version = "1.1.0"
        self.session_id = datetime.datetime.now().strftime("traj_%Y%m%d_%H%M%S")
        self.save_dir = Path(base_dir) / self.session_id
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.text_log_path = self.save_dir / "text_obs.jsonl"
        self.save_env_states = True
        self.env_states_dir = self.save_dir / "env_states"
        if self.save_env_states:
            self.env_states_dir.mkdir(parents=True, exist_ok=True)

        self.env_name = env_name
        self.env_params_digest = env_params_digest
        self.repo_commit = repo_commit
        self.created_at = datetime.datetime.now().isoformat()

        self.obs_vectors = []
        self.raw_texts = []
        self.episode_ids = []
        self.action_ids = []
        self.rewards = []
        self.dones = []
        self.timestamps = []
        self.env_state_paths = []

        print(f"Recording full trajectory to: {self.save_dir}")

    def record_timestep(
        self,
        obs,
        env_state,
        episode_id: int,
        action_id=None,
        reward=None,
        done=None,
    ):
        t = len(self.obs_vectors)
        obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        raw_text_obs = render_craftax_text(env_state)
        timestamp = datetime.datetime.now().isoformat()

        self.obs_vectors.append(obs_vec)
        self.raw_texts.append(raw_text_obs)
        self.episode_ids.append(int(episode_id))
        self.action_ids.append(-1 if action_id is None else int(action_id))
        self.rewards.append(np.nan if reward is None else float(reward))
        self.dones.append(False if done is None else bool(done))
        self.timestamps.append(timestamp)
        if self.save_env_states:
            rel_path = f"env_states/t_{t:05d}.pbz2"
            save_compressed_pickle(str(self.save_dir / "env_states" / f"t_{t:05d}"), env_state)
            self.env_state_paths.append(rel_path)
        else:
            self.env_state_paths.append("")

        entry = {
            "t": t,
            "timestamp": timestamp,
            "episode_id": int(episode_id),
            "action_id": None if action_id is None else int(action_id),
            "action_name": _action_name(action_id),
            "reward": None if reward is None else float(reward),
            "done": None if done is None else bool(done),
            "raw_text_obs": raw_text_obs,
            "env_state_path": self.env_state_paths[t] if self.save_env_states else None,
        }
        with self.text_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def finalize(self):
        num_timesteps = len(self.obs_vectors)
        if num_timesteps == 0:
            print("No timesteps recorded. Nothing to save.")
            return

        obs_array = np.stack(self.obs_vectors, axis=0).astype(np.float32)
        np.save(self.save_dir / "obs_vectors.npy", obs_array)

        np.savez_compressed(
            self.save_dir / "trajectory.npz",
            episode_ids=np.asarray(self.episode_ids, dtype=np.int32),
            action_ids=np.asarray(self.action_ids, dtype=np.int32),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            dones=np.asarray(self.dones, dtype=np.bool_),
            timestamps=np.asarray(self.timestamps, dtype=object),
        )

        paired_trajectory = [
            {
                "t": t,
                "obs_vector": self.obs_vectors[t],
                "raw_text_obs": self.raw_texts[t],
                "action_id": None if self.action_ids[t] < 0 else int(self.action_ids[t]),
                "action_name": _action_name(self.action_ids[t]) if self.action_ids[t] >= 0 else None,
                "reward": None if np.isnan(self.rewards[t]) else float(self.rewards[t]),
                "done": bool(self.dones[t]) if self.action_ids[t] >= 0 else None,
                "env_state_path": self.env_state_paths[t] if self.save_env_states else None,
            }
            for t in range(num_timesteps)
        ]
        save_compressed_pickle(
            str(self.save_dir / "trajectory_pairs"),
            paired_trajectory,
        )

        metadata = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "saved_at": datetime.datetime.now().isoformat(),
            "env_name": self.env_name,
            "env_params_digest": self.env_params_digest,
            "repo_commit": self.repo_commit,
            "num_timesteps": int(num_timesteps),
            "obs_dim": int(obs_array.shape[1]),
            "notes": (
                "Each timestep t stores one state (obs_vector, raw_text_obs). "
                "Action/reward/done at timestep t are aligned to that same state (state_t, action_t, reward_t, done_t). "
                "When recording stops by quit, a trailing final state is written with action_id=-1 and reward=NaN. "
                "When recording stops on done=True, no post-done reset state is written. "
                "If save_env_states=true, env_states/t_XXXXX.pbz2 contains exact EnvState snapshots."
            ),
            "paths": {
                "obs_vectors": "obs_vectors.npy",
                "text_obs": "text_obs.jsonl",
                "trajectory_arrays": "trajectory.npz",
                "trajectory_pairs": "trajectory_pairs.pbz2",
                "env_states_dir": ("env_states" if self.save_env_states else ""),
                "env_state_filename_pattern": ("t_00000.pbz2" if self.save_env_states else ""),
            },
        }
        (self.save_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True)
        )

        print("\nSaved full trajectory recording:")
        print(f"- Timesteps: {num_timesteps}")
        print(f"- Obs vectors: {self.save_dir / 'obs_vectors.npy'}")
        print(f"- Raw text log: {self.text_log_path}")
        print(f"- Paired trajectory: {self.save_dir / 'trajectory_pairs.pbz2'}")
        if self.save_env_states:
            print(f"- Env state snapshots: {self.save_dir / 'env_states'}")


def main(args):
    env_name = "Craftax-Symbolic-v1"
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env_params = env.default_params
    if args.god_mode:
        env_params = env_params.replace(god_mode=True)

    env_params_digest = hashlib.sha1(str(env_params).encode("utf-8")).hexdigest()
    repo_commit = _get_repo_commit()

    print("\n" + "=" * 58)
    print("CRAFTAX FULL TRAJECTORY RECORDER")
    print("=" * 58)
    print("Controls:")
    for key, action in KEY_MAPPING.items():
        print(f"{pygame.key.name(key)}: {action.name.lower()}")
    print("-" * 58)
    print("Recording mode: automatic at every timestep (from reset to quit).")
    print("=" * 58 + "\n")

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)

    pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN
    renderer = CraftaxRenderer(env, env_params, pixel_render_size=pixel_render_size)
    renderer.render(env_state)

    step_fn = jax.jit(env.step)
    recorder = TrajectoryRecorder(
        base_dir=args.output_dir,
        env_name=env_name,
        env_params_digest=env_params_digest,
        repo_commit=repo_commit,
    )

    episode_id = 0
    terminated_by_done = False

    clock = pygame.time.Clock()
    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress(env_state)
        if action is not None:
            obs_before = obs
            state_before = env_state
            episode_id_before = episode_id

            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.achievements
            obs, env_state, reward, done, _info = step_fn(
                _rng, env_state, action, env_params
            )

            recorder.record_timestep(
                obs=obs_before,
                env_state=state_before,
                episode_id=episode_id_before,
                action_id=action,
                reward=reward,
                done=done,
            )

            if done and not args.continue_after_done:
                terminated_by_done = True
                print("done=True detected; stopping recorder before post-reset state.")
                break

            if done:
                episode_id += 1

            new_achievements = env_state.achievements
            print_new_achievements(old_achievements, new_achievements)
            if reward > 0.8:
                print(f"Reward: {reward}\n")

            renderer.render(env_state)

        renderer.update()
        clock.tick(args.fps)

    # Write trailing state only if we stopped by user quit (not done-triggered stop).
    if not terminated_by_done:
        recorder.record_timestep(obs=obs, env_state=env_state, episode_id=episode_id)
    recorder.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play Craftax and record full (obs_vector, raw_text_obs) trajectory."
    )
    parser.add_argument("--god_mode", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--continue_after_done",
        action="store_true",
        help=(
            "Keep recording after done/reset boundaries. "
            "By default, recording stops at first done=True."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="play_data/trajectory_records",
        help="Base directory where session folders are written.",
    )
    args = parser.parse_args()

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
