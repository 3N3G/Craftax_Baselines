import argparse
import bz2
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax.constants import Action
from craftax.craftax.renderer import render_craftax_text
from craftax.craftax_env import make_craftax_env_from_name


ACTION_DIM = len(Action)
DEFAULT_HIGH_STAT = 9.0
DEFAULT_LOW_STAT = 1.0


def _replace_scalar(state, field_name: str, value: float):
    field_value = getattr(state, field_name)
    return state.replace(**{field_name: jnp.asarray(value, dtype=field_value.dtype)})


def _replace_inventory_scalar(state, field_name: str, value: float):
    inv = state.inventory
    inv_field_value = getattr(inv, field_name)
    new_inv = inv.replace(
        **{field_name: jnp.asarray(value, dtype=inv_field_value.dtype)}
    )
    return state.replace(inventory=new_inv)


def apply_pair(state, pair_name: str) -> Tuple[object, object, dict]:
    if pair_name == "health":
        low_value, high_value = DEFAULT_LOW_STAT, DEFAULT_HIGH_STAT
        low = _replace_scalar(state, "player_health", low_value)
        high = _replace_scalar(state, "player_health", high_value)
        meta = {"field": "player_health", "low_value": low_value, "high_value": high_value}
    elif pair_name == "food":
        low_value, high_value = DEFAULT_LOW_STAT, DEFAULT_HIGH_STAT
        low = _replace_scalar(state, "player_food", low_value)
        high = _replace_scalar(state, "player_food", high_value)
        meta = {"field": "player_food", "low_value": low_value, "high_value": high_value}
    elif pair_name == "drink":
        low_value, high_value = DEFAULT_LOW_STAT, DEFAULT_HIGH_STAT
        low = _replace_scalar(state, "player_drink", low_value)
        high = _replace_scalar(state, "player_drink", high_value)
        meta = {"field": "player_drink", "low_value": low_value, "high_value": high_value}
    elif pair_name == "energy":
        low_value, high_value = DEFAULT_LOW_STAT, DEFAULT_HIGH_STAT
        low = _replace_scalar(state, "player_energy", low_value)
        high = _replace_scalar(state, "player_energy", high_value)
        meta = {"field": "player_energy", "low_value": low_value, "high_value": high_value}
    elif pair_name == "wood":
        low_value, high_value = 0, 10
        low = _replace_inventory_scalar(state, "wood", low_value)
        high = _replace_inventory_scalar(state, "wood", high_value)
        meta = {"field": "inventory.wood", "low_value": low_value, "high_value": high_value}
    elif pair_name == "stone":
        low_value, high_value = 0, 10
        low = _replace_inventory_scalar(state, "stone", low_value)
        high = _replace_inventory_scalar(state, "stone", high_value)
        meta = {"field": "inventory.stone", "low_value": low_value, "high_value": high_value}
    else:
        raise ValueError(f"Unsupported pair type: {pair_name}")

    return low, high, meta


def collect_states(env, env_params, num_states: int, seed: int, max_steps: int) -> List[object]:
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    _, state = env.reset(reset_rng, env_params)

    states = []
    for _ in range(max_steps):
        if len(states) >= num_states:
            break
        states.append(state)
        rng, step_rng = jax.random.split(rng)
        action = int(np.random.randint(0, ACTION_DIM))
        _, state, _, _, _ = env.step(step_rng, state, action, env_params)
    return states


def save_compressed_pickle(path: Path, obj):
    with bz2.BZ2File(path, "wb") as f:
        pickle.dump(obj, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paired Craftax EnvState probes that differ in one key factor."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_states", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["health", "food", "drink", "energy", "wood", "stone"],
        choices=["health", "food", "drink", "energy", "wood", "stone"],
    )
    parser.add_argument(
        "--save_text",
        action="store_true",
        help="Also write raw rendered text observations for each low/high state.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    vectors_dir = output_dir / "vectors"
    states_dir = output_dir / "states"
    text_dir = output_dir / "text"
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)
    if args.save_text:
        text_dir.mkdir(parents=True, exist_ok=True)

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    base_states = collect_states(
        env=env,
        env_params=env_params,
        num_states=args.num_states,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    print(f"Collected {len(base_states)} base states.")

    metadata_path = output_dir / "pairs.jsonl"
    pair_count = 0
    with metadata_path.open("w") as f:
        for base_idx, base_state in enumerate(base_states):
            for pair_name in args.pairs:
                low_state, high_state, pair_meta = apply_pair(base_state, pair_name)
                obs_low = np.asarray(env.get_obs(low_state), dtype=np.float32)
                obs_high = np.asarray(env.get_obs(high_state), dtype=np.float32)

                pair_id = f"pair_{pair_count:06d}"
                np.savez_compressed(
                    vectors_dir / f"{pair_id}.npz",
                    obs_low=obs_low,
                    obs_high=obs_high,
                    base_index=np.asarray(base_idx, dtype=np.int32),
                )

                save_compressed_pickle(states_dir / f"{pair_id}_low.pbz2", low_state)
                save_compressed_pickle(states_dir / f"{pair_id}_high.pbz2", high_state)

                if args.save_text:
                    (text_dir / f"{pair_id}_low.txt").write_text(render_craftax_text(low_state))
                    (text_dir / f"{pair_id}_high.txt").write_text(render_craftax_text(high_state))

                row = {
                    "pair_id": pair_id,
                    "base_index": base_idx,
                    "pair_name": pair_name,
                    "expected_value_relation": "high > low",
                    **pair_meta,
                }
                f.write(json.dumps(row) + "\n")
                pair_count += 1

    summary = {
        "output_dir": str(output_dir),
        "num_base_states": len(base_states),
        "pairs_per_state": len(args.pairs),
        "total_pairs": pair_count,
        "metadata_file": str(metadata_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
