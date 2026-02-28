#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def latest_resume_payload(checkpoint_dir: str) -> Dict[str, Any]:
    latest_meta = Path(checkpoint_dir) / "latest_resume.json"
    if not latest_meta.exists():
        return {}
    with latest_meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    ckpt_path = meta.get("checkpoint_path")
    if not ckpt_path or not os.path.exists(ckpt_path):
        return {}
    with open(ckpt_path, "rb") as f:
        return pickle.load(f)


def infer_score(payload: Dict[str, Any]) -> float:
    tail = payload.get("episode_returns_tail", [])
    if not tail:
        return float("-inf")
    return float(sum(float(x) for x in tail) / len(tail))


def submit_continuation(entry: Dict[str, Any], promote_target: int, job_suffix: str):
    env = os.environ.copy()
    env["ENVS"] = str(entry["envs"])
    env["TARGET_TIMESTEPS"] = str(promote_target)
    env["SKIP_N"] = str(entry["skip_n"])
    env["LAYER"] = str(entry["layer"])
    env["TOKENS"] = str(entry["tokens"])
    env["NUM_STEPS"] = str(entry["num_steps"])
    env["CHECKPOINT_EVERY_STEPS"] = str(entry["checkpoint_every_steps"])
    env["POLICY_SAVE_DIR"] = str(entry["policy_save_dir"])
    env["CHECKPOINT_DIR"] = str(entry["checkpoint_dir"])
    env["RESUME_FROM"] = str(entry["checkpoint_dir"])
    env["RUN_NAME"] = str(entry["run_name"])
    env["JOB_NAME"] = f"{entry['job_name']}_{job_suffix}"
    env["HIDDEN_POOLING"] = str(entry["hidden_pooling"])
    env["HIDDEN_POOLING_K"] = str(entry["hidden_pooling_k"])
    env["TEMPERATURE"] = str(entry["temperature"])
    env["SAVE_TRAJ_ONLINE"] = str(entry.get("save_traj_online", 0))
    env["TRAJ_SAVE_DIR"] = str(entry.get("traj_save_dir", ""))
    env["TRAJ_SAVE_EVERY_UPDATES"] = str(entry.get("traj_save_every_updates", 50))
    env["TRAJ_FREE_SPACE_MIN_GB"] = str(entry.get("traj_free_space_min_gb", 150))
    env["TRAJ_SCHEMA"] = str(entry.get("traj_schema", "minimal_core"))

    cmd = ["bash", "scripts/shell/submit_online_rl_hidden_chain.sh"]
    out = subprocess.check_output(cmd, env=env, text=True).strip()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--promote-target", type=int, default=300_000_000)
    parser.add_argument("--report-path", default=None)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        entries: List[Dict[str, Any]] = json.load(f)

    scored: List[Dict[str, Any]] = []
    for entry in entries:
        payload = latest_resume_payload(entry["checkpoint_dir"])
        if not payload:
            score = float("-inf")
            total_steps = 0
        else:
            score = infer_score(payload)
            total_steps = int(payload.get("total_steps", 0))
        scored.append(
            {
                **entry,
                "score": score,
                "total_steps": total_steps,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    selected = [s for s in scored if s["score"] != float("-inf")][: args.top_k]
    print("Selector ranking:")
    for idx, row in enumerate(scored, start=1):
        print(
            f"  {idx:02d}. run={row['run_name']} "
            f"score={row['score']:.4f} step={row['total_steps']}"
        )

    submit_logs = []
    for idx, entry in enumerate(selected, start=1):
        out = submit_continuation(entry, args.promote_target, f"promote{idx}")
        submit_logs.append({"run_name": entry["run_name"], "submit_output": out})
        print(f"Submitted promotion for {entry['run_name']}: {out}")

    report = {
        "manifest": args.manifest,
        "top_k": args.top_k,
        "promote_target": args.promote_target,
        "ranked": scored,
        "selected": selected,
        "submissions": submit_logs,
    }
    report_path = (
        args.report_path
        if args.report_path
        else str(Path(args.manifest).with_suffix(".promotion_report.json"))
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"Wrote promotion report: {report_path}")


if __name__ == "__main__":
    main()
