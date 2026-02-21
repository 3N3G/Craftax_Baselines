#!/usr/bin/env python3
"""Parse online RL SPS diagnostics logs and print a compact summary table."""

import glob
import os
import re
import sys
from typing import Optional


DONE_RE = re.compile(
    r"Done\. SPS:\s*([0-9,]+(?:\.[0-9]+)?)"
    r"(?:,\s*LLM calls:\s*([0-9,]+))?"
    r"(?:,\s*Return:\s*([0-9.+-]+))?"
)
SUMMARY_RE = re.compile(
    r"SUMMARY .*?extract_ms_mean=([0-9.]+).*?text_ms_mean=([0-9.]+).*?"
    r"llm_ms_mean=([0-9.]+).*?approx_sps=([0-9.]+)"
)


def _as_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    return float(s.replace(",", ""))


def _as_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    return int(s.replace(",", ""))


def parse_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    row = {"file": os.path.basename(path), "kind": "unknown"}
    m = DONE_RE.search(text)
    if m:
        row["kind"] = "train"
        row["sps"] = _as_float(m.group(1))
        row["llm_calls"] = _as_int(m.group(2))
        row["ret"] = _as_float(m.group(3))

    m2 = SUMMARY_RE.search(text)
    if m2:
        row["kind"] = "extract"
        row["extract_ms_mean"] = float(m2.group(1))
        row["text_ms_mean"] = float(m2.group(2))
        row["llm_ms_mean"] = float(m2.group(3))
        row["approx_sps"] = float(m2.group(4))

    return row


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: parse_online_rl_sps_diag.py <log_dir>")
        return 2

    log_dir = sys.argv[1]
    if not os.path.isdir(log_dir):
        print(f"ERROR: not a directory: {log_dir}")
        return 2

    paths = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not paths:
        print(f"No .log files found under {log_dir}")
        return 1

    rows = [parse_file(p) for p in paths]

    print("=" * 100)
    print(f"SPS Diagnostics Summary: {log_dir}")
    print("=" * 100)

    print("\n[Training runs]")
    print(f"{'file':40s} {'sps':>10s} {'llm_calls':>12s} {'return':>10s}")
    for r in rows:
        if r.get("kind") != "train":
            continue
        sps = f"{r.get('sps', 0):.2f}" if r.get("sps") is not None else "-"
        llm_calls = str(r.get("llm_calls")) if r.get("llm_calls") is not None else "-"
        ret = f"{r.get('ret', 0):.2f}" if r.get("ret") is not None else "-"
        print(f"{r['file'][:40]:40s} {sps:>10s} {llm_calls:>12s} {ret:>10s}")

    print("\n[Extractor microbenchmark]")
    print(
        f"{'file':40s} {'extract_ms':>12s} {'text_ms':>10s} "
        f"{'llm_ms':>10s} {'approx_sps':>12s}"
    )
    for r in rows:
        if r.get("kind") != "extract":
            continue
        print(
            f"{r['file'][:40]:40s} "
            f"{r.get('extract_ms_mean', 0):12.2f} "
            f"{r.get('text_ms_mean', 0):10.2f} "
            f"{r.get('llm_ms_mean', 0):10.2f} "
            f"{r.get('approx_sps', 0):12.2f}"
        )

    print("\nRaw log files:")
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
