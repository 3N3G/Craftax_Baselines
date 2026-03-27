#!/usr/bin/env python3
"""
Render a single "big table" of hidden-state latency across model/backends.

This script scans latency benchmark outputs under:
  analysis/future_imagination_latency_bench/<run_dir>/<run_label>/{summary.json,run_meta.json}

It then maps discovered results onto a model matrix file and emits:
  - markdown table
  - csv table
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def _safe_float(v: object) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: object) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None


@dataclass
class RunSummary:
    parent_run_dir: Path
    run_label: str
    model: str
    provider: str
    summary_rows: List[Dict[str, object]]
    meta: Dict[str, object]


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover(root: Path) -> List[RunSummary]:
    found: List[RunSummary] = []
    if not root.exists():
        return found
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for child in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
            summary_path = child / "summary.json"
            meta_path = child / "run_meta.json"
            if not summary_path.exists():
                continue
            try:
                summary_rows = _load_json(summary_path)
                if not isinstance(summary_rows, list):
                    continue
                meta = _load_json(meta_path) if meta_path.exists() else {}
                if not isinstance(meta, dict):
                    meta = {}
            except Exception:
                continue
            model = str(meta.get("model", "")) or str(
                summary_rows[0].get("model", "") if summary_rows else ""
            )
            provider = str(meta.get("provider", ""))
            found.append(
                RunSummary(
                    parent_run_dir=run_dir,
                    run_label=child.name,
                    model=model,
                    provider=provider,
                    summary_rows=[r for r in summary_rows if isinstance(r, dict)],
                    meta=meta,
                )
            )
    return found


def _weighted_avg(rows: List[Dict[str, object]], key: str, weight_key: str) -> Optional[float]:
    num = 0.0
    den = 0.0
    for r in rows:
        v = _safe_float(r.get(key))
        w = _safe_float(r.get(weight_key))
        if v is None or w is None or w <= 0:
            continue
        num += v * w
        den += w
    if den <= 0:
        return None
    return num / den


def _fmt(v: object, nd: int = 4) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def _pick_best(candidates: List[RunSummary]) -> Optional[RunSummary]:
    if not candidates:
        return None
    # Prefer latest timestamped parent folder (lexicographic works for YYYYmmdd_HHMMSS prefix)
    candidates = sorted(candidates, key=lambda x: x.parent_run_dir.name)
    return candidates[-1]


def _match_matrix_entry(
    entry: Dict[str, object],
    discovered: List[RunSummary],
) -> Optional[RunSummary]:
    run_label = str(entry.get("run_label", ""))
    model_match = str(entry.get("model_match", ""))
    by_label = [d for d in discovered if d.run_label == run_label]
    latest_by_label = _pick_best(by_label)
    subset = by_label
    if model_match:
        subset = [d for d in subset if model_match in d.model]
    matched_best = _pick_best(subset)

    # Some runs (notably qwen3.5 compat mirrors) intentionally benchmark a local
    # rewritten model path, so matrix model_match may only match older artifacts.
    # Prefer the newest artifact for the run label when it is newer than the
    # best model_match candidate.
    if latest_by_label is not None:
        if matched_best is None:
            return latest_by_label
        if latest_by_label.parent_run_dir.name > matched_best.parent_run_dir.name:
            return latest_by_label
    return matched_best


def _aggregate_row(
    entry: Dict[str, object],
    run: Optional[RunSummary],
    root: Path,
) -> Dict[str, object]:
    base = {
        "name": str(entry.get("name", "")),
        "category": str(entry.get("category", "")),
        "backend": str(entry.get("backend", "")),
        "run_label": str(entry.get("run_label", "")),
        "model": "",
        "provider": "",
        "benchmark_mode": "",
        "status": "missing",
        "total_calls": "",
        "successful_calls": "",
        "error_calls": "",
        "state_prompt_tokens": "",
        "state_completion_tokens": "",
        "state_request_latency_seconds": "",
        "state_request_calls_per_second": "",
        "state_hidden_ready_latency_seconds": "",
        "state_hidden_ready_calls_per_second": "",
        "history_prompt_tokens": "",
        "history_completion_tokens": "",
        "history_request_latency_seconds": "",
        "history_request_calls_per_second": "",
        "history_hidden_ready_latency_seconds": "",
        "history_hidden_ready_calls_per_second": "",
        "source_run_dir": "",
    }
    if run is None:
        return base

    rows = run.summary_rows
    total_calls = sum(_safe_int(r.get("calls")) or 0 for r in rows)
    ok_calls = sum(_safe_int(r.get("ok_calls")) or 0 for r in rows)
    err_calls = sum(_safe_int(r.get("error_calls")) or 0 for r in rows)
    by_variant = {
        str(r.get("variant", "")): r
        for r in rows
        if isinstance(r, dict) and str(r.get("variant", ""))
    }

    status = "ok" if ok_calls > 0 else "error"
    source = str(run.parent_run_dir / run.run_label)
    try:
        source = str((run.parent_run_dir / run.run_label).relative_to(root.parent))
    except Exception:
        pass

    base.update(
        {
            "model": run.model,
            "provider": run.provider,
            "benchmark_mode": str(run.meta.get("benchmark_mode", "")),
            "status": status,
            "total_calls": total_calls,
            "successful_calls": ok_calls,
            "error_calls": err_calls,
            "source_run_dir": source,
        }
    )

    variant_specs = {
        "state": "predict_state_only",
        "history": "predict_history_k5",
    }
    for prefix, variant_name in variant_specs.items():
        row = by_variant.get(variant_name)
        if not row:
            continue
        base[f"{prefix}_prompt_tokens"] = _safe_float(row.get("avg_prompt_tokens"))
        base[f"{prefix}_completion_tokens"] = _safe_float(
            row.get("avg_completion_tokens")
        )
        base[f"{prefix}_request_latency_seconds"] = _safe_float(
            row.get("avg_request_latency_seconds")
        )
        base[f"{prefix}_request_calls_per_second"] = _safe_float(
            row.get("avg_requests_per_second")
        )
        base[f"{prefix}_hidden_ready_latency_seconds"] = _safe_float(
            row.get("avg_end_to_end_latency_seconds")
        )
        base[f"{prefix}_hidden_ready_calls_per_second"] = _safe_float(
            row.get("avg_end_to_end_responses_per_second")
        )
    return base


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "name",
        "category",
        "backend",
        "run_label",
        "model",
        "provider",
        "benchmark_mode",
        "status",
        "total_calls",
        "successful_calls",
        "error_calls",
        "state_prompt_tokens",
        "state_completion_tokens",
        "state_request_latency_seconds",
        "state_request_calls_per_second",
        "state_hidden_ready_latency_seconds",
        "state_hidden_ready_calls_per_second",
        "history_prompt_tokens",
        "history_completion_tokens",
        "history_request_latency_seconds",
        "history_request_calls_per_second",
        "history_hidden_ready_latency_seconds",
        "history_hidden_ready_calls_per_second",
        "source_run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = dict(row)
            for k in (
                "state_prompt_tokens",
                "state_completion_tokens",
                "state_request_latency_seconds",
                "state_request_calls_per_second",
                "state_hidden_ready_latency_seconds",
                "state_hidden_ready_calls_per_second",
                "history_prompt_tokens",
                "history_completion_tokens",
                "history_request_latency_seconds",
                "history_request_calls_per_second",
                "history_hidden_ready_latency_seconds",
                "history_hidden_ready_calls_per_second",
            ):
                out[k] = _fmt(out.get(k), nd=6)
            w.writerow(out)


def _write_md(path: Path, rows: List[Dict[str, object]], root: Path, matrix_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Hidden-State Latency Model Table")
    lines.append("")
    lines.append(f"- benchmark_root: `{root}`")
    lines.append(f"- model_matrix: `{matrix_path}`")
    lines.append("- `request_avg_s`: average latency for the model/backend call itself.")
    lines.append("- `request_avg_responses_per_second`: inverse of `request_avg_s`.")
    lines.append("- `hidden_ready_avg_s`: end-to-end latency until the hidden vector is readable by the benchmark.")
    lines.append("- `hidden_ready_avg_responses_per_second`: inverse of `hidden_ready_avg_s`.")
    lines.append("")
    lines.append(
        "| model_name | category | backend | benchmark_mode | status | state_prompt_tokens | state_output_tokens | state_request_avg_s | state_request_avg_responses_per_second | state_hidden_ready_avg_s | state_hidden_ready_avg_responses_per_second | history_prompt_tokens | history_output_tokens | history_request_avg_s | history_request_avg_responses_per_second | history_hidden_ready_avg_s | history_hidden_ready_avg_responses_per_second | total_calls | successful_calls | error_calls | source_run_dir |"
    )
    lines.append(
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for r in rows:
        lines.append(
            "| {name} | {category} | {backend} | {benchmark_mode} | {status} | {state_prompt_tokens} | {state_completion_tokens} | {state_request_latency_seconds} | {state_request_calls_per_second} | {state_hidden_ready_latency_seconds} | {state_hidden_ready_calls_per_second} | {history_prompt_tokens} | {history_completion_tokens} | {history_request_latency_seconds} | {history_request_calls_per_second} | {history_hidden_ready_latency_seconds} | {history_hidden_ready_calls_per_second} | {calls} | {ok} | {err} | {src} |".format(
                name=r.get("name", ""),
                category=r.get("category", ""),
                backend=r.get("backend", ""),
                benchmark_mode=r.get("benchmark_mode", ""),
                status=r.get("status", ""),
                state_prompt_tokens=_fmt(r.get("state_prompt_tokens"), 1),
                state_completion_tokens=_fmt(r.get("state_completion_tokens"), 1),
                state_request_latency_seconds=_fmt(
                    r.get("state_request_latency_seconds"), 4
                ),
                state_request_calls_per_second=_fmt(
                    r.get("state_request_calls_per_second"), 4
                ),
                state_hidden_ready_latency_seconds=_fmt(
                    r.get("state_hidden_ready_latency_seconds"), 4
                ),
                state_hidden_ready_calls_per_second=_fmt(
                    r.get("state_hidden_ready_calls_per_second"), 4
                ),
                history_prompt_tokens=_fmt(r.get("history_prompt_tokens"), 1),
                history_completion_tokens=_fmt(
                    r.get("history_completion_tokens"), 1
                ),
                history_request_latency_seconds=_fmt(
                    r.get("history_request_latency_seconds"), 4
                ),
                history_request_calls_per_second=_fmt(
                    r.get("history_request_calls_per_second"), 4
                ),
                history_hidden_ready_latency_seconds=_fmt(
                    r.get("history_hidden_ready_latency_seconds"), 4
                ),
                history_hidden_ready_calls_per_second=_fmt(
                    r.get("history_hidden_ready_calls_per_second"), 4
                ),
                calls=r.get("total_calls", ""),
                ok=r.get("successful_calls", ""),
                err=r.get("error_calls", ""),
                src=r.get("source_run_dir", ""),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmark-root",
        default="analysis/future_imagination_latency_bench",
        help="Root directory containing benchmark run folders.",
    )
    p.add_argument(
        "--model-matrix",
        default="configs/future_imagination/hidden_latency_model_matrix.json",
        help="JSON file listing expected models/run labels.",
    )
    p.add_argument(
        "--output-md",
        default="analysis/future_imagination_latency_bench/hidden_latency_model_table.md",
        help="Output markdown path.",
    )
    p.add_argument(
        "--output-csv",
        default="analysis/future_imagination_latency_bench/hidden_latency_model_table.csv",
        help="Output csv path.",
    )
    args = p.parse_args()

    root = Path(args.benchmark_root)
    matrix_path = Path(args.model_matrix)
    matrix = _load_json(matrix_path)
    if not isinstance(matrix, list):
        raise ValueError(f"Expected list in model matrix: {matrix_path}")
    matrix_entries = [m for m in matrix if isinstance(m, dict)]

    discovered = _discover(root)
    rows: List[Dict[str, object]] = []
    for entry in matrix_entries:
        matched = _match_matrix_entry(entry, discovered)
        rows.append(_aggregate_row(entry, matched, root))

    out_md = Path(args.output_md)
    out_csv = Path(args.output_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_md(out_md, rows, root, matrix_path)
    _write_csv(out_csv, rows)

    print(f"rows={len(rows)}")
    print(f"output_md={out_md}")
    print(f"output_csv={out_csv}")


if __name__ == "__main__":
    main()
