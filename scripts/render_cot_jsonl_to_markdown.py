#!/usr/bin/env python3
"""Render CoT JSONL logs to markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def safe_block(text: object) -> str:
    if text is None:
        return ""
    return str(text).replace("```", "```\u200b")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to source JSONL file")
    parser.add_argument("--output", required=True, help="Path to output markdown file")
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.output)

    entries = []
    with src.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    lines: list[str] = []
    lines.append("# CoT Log Render")
    lines.append("")
    lines.append(f"- Source: `{src}`")
    lines.append(f"- Entries: `{len(entries)}`")
    lines.append("")

    for idx, rec in enumerate(entries, 1):
        update_idx = rec.get("update_idx", idx)
        timestep = rec.get("timestep", "n/a")
        timestamp = rec.get("timestamp", "n/a")
        llm_calls = rec.get("llm_calls", "n/a")
        prompt_variant = rec.get("prompt_variant", "n/a")

        lines.append(f"## Entry {idx}: update {update_idx}, timestep {timestep}")
        lines.append(f"- Timestamp: `{timestamp}`")
        lines.append(f"- LLM calls: `{llm_calls}`")
        lines.append(f"- Prompt variant: `{prompt_variant}`")

        prompt_outline = rec.get("prompt_outline")
        if prompt_outline is not None:
            if isinstance(prompt_outline, str):
                outline_text = prompt_outline
            else:
                outline_text = json.dumps(prompt_outline, indent=2, ensure_ascii=False)
            lines.append("- Prompt outline:")
            lines.append("```text")
            lines.append(safe_block(outline_text))
            lines.append("```")

        samples = rec.get("samples", [])
        lines.append(f"- Samples: `{len(samples)}`")

        for s_idx, sample in enumerate(samples, 1):
            prompt = safe_block(sample.get("prompt", ""))
            response = safe_block(sample.get("response", ""))
            lines.append(f"### Sample {s_idx}")
            lines.append("#### Prompt")
            lines.append("```text")
            lines.append(prompt)
            lines.append("```")
            lines.append("#### Response")
            lines.append("```text")
            lines.append(response)
            lines.append("```")

        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
