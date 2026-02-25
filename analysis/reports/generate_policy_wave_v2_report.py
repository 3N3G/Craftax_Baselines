#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _safe_mean(x: List[float]) -> float:
    return float(sum(x) / len(x)) if x else 0.0


def _policy_rows_id(summary: Dict) -> List[Tuple[str, float, float, float]]:
    rows = []
    for key, payload in summary.get("results", {}).get("id", {}).items():
        agg = payload.get("aggregate", {})
        rows.append(
            (
                key,
                float(agg.get("return_mean_over_seeds", 0.0)),
                float(agg.get("achievement_mean_over_seeds", 0.0)),
                float(agg.get("llm_calls_mean_over_seeds", 0.0)),
            )
        )
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _ood_summaries(summary: Dict) -> List[Tuple[str, float, bool]]:
    rows = []
    ood = summary.get("results", {}).get("ood", {})
    for sid, payload in ood.items():
        policy_map = payload.get("policies", {})
        ret_means = []
        for pol_payload in policy_map.values():
            agg = pol_payload.get("aggregate", {})
            ret_means.append(float(agg.get("return_mean_over_seeds", 0.0)))
        rows.append((sid, _safe_mean(ret_means), bool(payload.get("diagnostic_only", False))))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _load_json_if_exists(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def build_report(summary: Dict, summary_path: Path) -> str:
    lines: List[str] = []
    lines.append("# Policy Wave v2 Report")
    lines.append("")
    lines.append(f"- Generated from: `{summary_path}`")
    lines.append(f"- Timestamp: `{summary.get('timestamp', 'unknown')}`")
    lines.append(f"- Tracks: `{', '.join(summary.get('tracks', []))}`")
    lines.append("")

    lines.append("## ID Gameplay")
    rows = _policy_rows_id(summary)
    if rows:
        lines.append("| Policy Variant | Return Mean | Achievement Mean | LLM Calls Mean |")
        lines.append("|---|---:|---:|---:|")
        for name, r, a, c in rows:
            lines.append(f"| `{name}` | {r:.3f} | {a:.3f} | {c:.2f} |")
    else:
        lines.append("No ID results found.")
    lines.append("")

    lines.append("## OOD Scenarios")
    ood_rows = _ood_summaries(summary)
    if ood_rows:
        lines.append("| Scenario | Avg Return Across Policies | Diagnostic Only |")
        lines.append("|---|---:|---:|")
        for sid, ret, diag in ood_rows:
            lines.append(f"| `{sid}` | {ret:.3f} | {str(diag).lower()} |")
    else:
        lines.append("No OOD results found.")
    lines.append("")

    lines.append("## Gameplay LLM")
    llm = summary.get("results", {}).get("gameplay_llm", {})
    if llm:
        lines.append("| Policy Variant | Return Mean | LLM Calls Mean |")
        lines.append("|---|---:|---:|")
        rows_llm = []
        for key, payload in llm.items():
            agg = payload.get("aggregate", {})
            rows_llm.append(
                (
                    key,
                    float(agg.get("return_mean_over_seeds", 0.0)),
                    float(agg.get("llm_calls_mean_over_seeds", 0.0)),
                )
            )
        rows_llm.sort(key=lambda x: x[1], reverse=True)
        for name, r, c in rows_llm:
            lines.append(f"| `{name}` | {r:.3f} | {c:.2f} |")
    else:
        lines.append("No gameplay_llm results found.")
    lines.append("")

    lines.append("## Value Battery")
    value = summary.get("results", {}).get("value", {})
    if value:
        lines.append(f"- Status: `{value.get('status', 'unknown')}`")
        for k in ["value_learning_json", "value_pairs_json", "value_td_json"]:
            if k in value:
                lines.append(f"- {k}: `{value[k]}`")
    else:
        lines.append("No value-battery result block found.")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report from policy_wave_v2 summary json")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument(
        "--output_md",
        type=str,
        default="analysis/reports/policy_wave_v2_report.md",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_json).expanduser().resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_json not found: {summary_path}")

    summary = json.loads(summary_path.read_text())
    report = build_report(summary, summary_path=summary_path)

    out_path = Path(args.output_md).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
