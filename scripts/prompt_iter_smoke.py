#!/usr/bin/env python3
"""Smoke runner for prompt_iter backend over the fixed 10-state set."""

from __future__ import annotations

import argparse
import json
from statistics import mean

from scripts import prompt_iter_backend as backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prompt-iteration smoke over fixed states")
    parser.add_argument("--variant", default="default", choices=["default", "future_based", "future_based_opt"])
    parser.add_argument("--server-url", default=backend.DEFAULT_VLLM_URL)
    parser.add_argument("--model-name", default=backend.DEFAULT_VLLM_MODEL)
    parser.add_argument("--model-id", default=backend.DEFAULT_MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-json", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    states = backend.load_fixed_states()
    sections = backend.default_prompt_sections(args.variant)
    results = backend.run_batch(
        states,
        sections,
        server_url=args.server_url,
        model_name=args.model_name,
        model_id=args.model_id,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop_sequences=sections.stop_sequences,
    )

    payload = {
        "count": len(results),
        "variant": args.variant,
        "state_ids": [r["state_id"] for r in results],
        "latency_mean_s": mean(float(r["latency_s"]) for r in results) if results else 0.0,
        "latency_max_s": max(float(r["latency_s"]) for r in results) if results else 0.0,
        "response_chars": {r["state_id"]: len(r.get("response_text", "")) for r in results},
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"wrote {args.output_json}")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
