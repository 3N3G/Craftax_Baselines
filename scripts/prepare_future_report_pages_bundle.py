#!/usr/bin/env python3
"""Prepare a future-imagination combined report for static hosting (GitHub Pages, Netlify, etc.)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


VIEWER_DATA_RE = re.compile(
    r"(<script id='viewer-data' type='application/json'>)(.*?)(</script>)",
    flags=re.DOTALL,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", type=Path, required=True, help="Directory with combined_unabridged_report.html")
    p.add_argument("--frame-dir", type=Path, required=True, help="Directory containing t_XXXXX.png frames")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for hosted bundle")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it already exists",
    )
    p.add_argument(
        "--index-alias",
        action="store_true",
        default=True,
        help="Also write index.html as an alias to combined_unabridged_report.html content",
    )
    return p


def _load_viewer_payload(html_text: str) -> tuple[dict, tuple[int, int], tuple[str, str, str]]:
    m = VIEWER_DATA_RE.search(html_text)
    if not m:
        raise ValueError("viewer-data JSON block not found in HTML")
    prefix, payload_text, suffix = m.group(1), m.group(2), m.group(3)
    payload = json.loads(payload_text)
    return payload, m.span(2), (prefix, payload_text, suffix)


def _rewrite_and_copy_frames(payload: dict, frame_dir: Path, out_frames_dir: Path) -> tuple[int, int]:
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    copied_names: set[str] = set()

    for state in payload.get("states", []):
        raw_path = str(state.get("frame_path", "")).strip()
        if not raw_path:
            continue

        parts = urlsplit(raw_path)
        original_path = parts.path
        basename = Path(original_path).name
        src = frame_dir / basename
        dst = out_frames_dir / basename

        if src.exists():
            if basename not in copied_names:
                shutil.copy2(src, dst)
                copied_names.add(basename)
                copied += 1
            new_path = urlunsplit(("", "", f"frames/{basename}", parts.query, ""))
            state["frame_path"] = new_path
        else:
            missing += 1
            state["frame_path"] = ""

    return copied, missing


def main() -> None:
    args = build_parser().parse_args()

    report_dir = args.report_dir.resolve()
    frame_dir = args.frame_dir.resolve()
    output_dir = args.output_dir.resolve()

    html_in = report_dir / "combined_unabridged_report.html"
    md_in = report_dir / "combined_unabridged_report.md"
    records_in = report_dir / "combined_records.jsonl"

    if not html_in.exists():
        raise FileNotFoundError(f"Missing HTML report: {html_in}")
    if not frame_dir.exists():
        raise FileNotFoundError(f"Missing frame dir: {frame_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir exists: {output_dir} (pass --overwrite)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_text = html_in.read_text(encoding="utf-8")
    payload, payload_span, _ = _load_viewer_payload(html_text)

    copied, missing = _rewrite_and_copy_frames(payload, frame_dir=frame_dir, out_frames_dir=output_dir / "frames")

    new_payload_text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    html_out_text = html_text[: payload_span[0]] + new_payload_text + html_text[payload_span[1] :]

    html_out = output_dir / "combined_unabridged_report.html"
    html_out.write_text(html_out_text, encoding="utf-8")

    if args.index_alias:
        (output_dir / "index.html").write_text(html_out_text, encoding="utf-8")

    if md_in.exists():
        shutil.copy2(md_in, output_dir / md_in.name)
    if records_in.exists():
        shutil.copy2(records_in, output_dir / records_in.name)

    meta = {
        "source_report_dir": str(report_dir),
        "source_frame_dir": str(frame_dir),
        "output_dir": str(output_dir),
        "frames_copied": copied,
        "frames_missing": missing,
    }
    (output_dir / "bundle_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")

    print(f"output_dir={output_dir}")
    print(f"frames_copied={copied}")
    print(f"frames_missing={missing}")
    print(f"entrypoint={output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
