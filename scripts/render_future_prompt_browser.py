#!/usr/bin/env python3
"""Render full saved prompt files from future-imagination runs into markdown."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _safe_block(text: str) -> str:
    return text.replace("```", "```\u200b")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True, help="Future imagination run dir with records.jsonl")
    p.add_argument("--output", type=Path, default=None, help="Output markdown file (default: <run-dir>/prompt_browser.md)")
    p.add_argument("--max-items", type=int, default=0, help="If >0, limit rendered prompts to this many entries")
    p.add_argument("--include-response", action="store_true", help="Include model response text below each prompt")
    p.add_argument("--run-id", type=str, default="", help="Optional filter for one run_id")
    p.add_argument("--t-min", type=int, default=None)
    p.add_argument("--t-max", type=int, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    records_path = run_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing records.jsonl: {records_path}")

    output_path = args.output.resolve() if args.output else (run_dir / "prompt_browser.md")
    records = _read_jsonl(records_path)

    filtered: List[Dict[str, Any]] = []
    for rec in records:
        run_id = str(rec.get("run_id", ""))
        t = _safe_int(rec.get("t"))
        if not rec.get("prompt_path"):
            continue
        if args.run_id and run_id != args.run_id:
            continue
        if t is None:
            continue
        if args.t_min is not None and t < args.t_min:
            continue
        if args.t_max is not None and t > args.t_max:
            continue
        filtered.append(rec)

    filtered.sort(key=lambda r: (_safe_int(r.get("t")) or -1, str(r.get("run_id", ""))))
    if args.max_items > 0:
        filtered = filtered[: args.max_items]

    lines: List[str] = []
    lines.append("# Prompt Browser")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat()}`")
    lines.append(f"- Source run dir: `{run_dir}`")
    lines.append(f"- Entries rendered: `{len(filtered)}`")
    if args.run_id:
        lines.append(f"- Filter run_id: `{args.run_id}`")
    if args.t_min is not None or args.t_max is not None:
        lines.append(f"- Filter t range: `{args.t_min}`..`{args.t_max}`")
    lines.append("")

    for idx, rec in enumerate(filtered, 1):
        t = _safe_int(rec.get("t"))
        run_id = str(rec.get("run_id", ""))
        status = str(rec.get("status", ""))
        latency = rec.get("latency_s")
        prompt_path = Path(str(rec.get("prompt_path"))).resolve()
        prompt_text = ""
        if prompt_path.exists():
            prompt_text = prompt_path.read_text(encoding="utf-8")
        else:
            prompt_text = f"[missing prompt file: {prompt_path}]"

        lines.append(f"## {idx}. t={t}, run_id={run_id}")
        lines.append(f"- status: `{status}`")
        lines.append(f"- latency_s: `{latency}`")
        lines.append(f"- prompt_path: `{prompt_path}`")
        lines.append("")
        lines.append("```text")
        lines.append(_safe_block(prompt_text))
        lines.append("```")
        lines.append("")

        if args.include_response:
            response_text = str(rec.get("response_text", ""))
            lines.append("<details><summary>Response</summary>")
            lines.append("")
            lines.append("```text")
            lines.append(_safe_block(response_text))
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
