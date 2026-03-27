#!/usr/bin/env python3
"""Thin wrapper for qualitative LLM reaction analysis on recorder bundles."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run bundle-only qualitative LLM reaction eval")
    parser.add_argument("--manifest", type=str, default=str(REPO_ROOT / "configs" / "eval" / "policy_wave_v2.yaml"))
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "analysis" / "policy_wave_v2" / "bundle_reactions"))
    parser.add_argument("--bundle_dirs", type=str, required=True)
    parser.add_argument("--llm_generation_tokens", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_episodes", type=int, default=128)
    parser.add_argument("--max_env_steps", type=int, default=80000)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_policy_wave.py"),
        "--manifest",
        args.manifest,
        "--tracks",
        "bundle",
        "--output_dir",
        args.output_dir,
        "--bundle_dirs",
        args.bundle_dirs,
        "--llm_generation_tokens",
        str(args.llm_generation_tokens),
        "--seeds",
        args.seeds,
        "--num_envs",
        str(args.num_envs),
        "--num_episodes",
        str(args.num_episodes),
        "--max_env_steps",
        str(args.max_env_steps),
    ]
    if args.no_wandb:
        cmd.append("--no_wandb")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
