#!/usr/bin/env python3
"""Render unabridged combined prompt/response reports across multiple future-imagination runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from utils.llm_prompts import filter_text_obs as _filter_text_obs
except Exception:
    _filter_text_obs = None

MAP_INTERESTING_PREFIX = "Map (interesting tiles only):"
_MAP_COORD_PREFIX_RE = re.compile(r"-?\d+\s*,\s*-?\d+\s*:")
_MAP_ENTRY_RE = re.compile(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*:\s*(.+?)\s*$")


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_block(text: str) -> str:
    return text.replace("```", "```\u200b")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory containing records.jsonl (repeatable)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for combined artifacts",
    )
    p.add_argument(
        "--timesteps",
        type=str,
        default="",
        help="Optional comma-separated timestep filter (e.g., 0,283,298)",
    )
    p.add_argument(
        "--include-errors",
        action="store_true",
        help="Include error records (default: only status=ok)",
    )
    p.add_argument(
        "--dedupe-by-timestep",
        action="store_true",
        help="Keep only one record per timestep (latest by timestamp, then order).",
    )
    p.add_argument(
        "--collapse-prompts",
        action="store_true",
        help="Deprecated: prompt blocks are collapsed by default.",
    )
    p.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Optional trajectory dir containing text_obs.jsonl for timeline viewer.",
    )
    p.add_argument(
        "--viewer-frame-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory with pixel frames named t_00000.png for the timeline viewer. "
            "Defaults to <trajectory-dir>/render_frames_bs16 when present."
        ),
    )
    p.add_argument(
        "--disable-viewer",
        action="store_true",
        help="Skip embedding the timeline viewer in HTML.",
    )
    return p


def _parse_timesteps(text: str) -> Optional[set[int]]:
    stripped = (text or "").strip()
    if not stripped:
        return None
    values: set[int] = set()
    for tok in stripped.split(","):
        tok = tok.strip()
        if not tok:
            continue
        values.add(int(tok))
    return values


def _load_records(run_dirs: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rd in run_dirs:
        rec_path = rd / "records.jsonl"
        if not rec_path.exists():
            raise FileNotFoundError(f"Missing records file: {rec_path}")
        rows = _read_jsonl(rec_path)
        for row in rows:
            row["_source_run_dir"] = str(rd)
            out.append(row)
    return out


def _record_sort_key(rec: Dict[str, Any]) -> Tuple[int, str]:
    t = _safe_int(rec.get("t"))
    ts = str(rec.get("timestamp", ""))
    return (t if t is not None else 10**9, ts)


def _dedupe_latest_by_timestep(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    for rec in sorted(records, key=_record_sort_key):
        t = _safe_int(rec.get("t"))
        if t is None:
            continue
        latest[t] = rec
    return [latest[t] for t in sorted(latest.keys())]


def _resolve_prompt_path(rec: Dict[str, Any]) -> Optional[Path]:
    prompt_path_raw = str(rec.get("prompt_path", "")).strip()
    if not prompt_path_raw:
        return None

    direct = Path(prompt_path_raw)
    if direct.exists():
        return direct

    # When records were produced on Babel, prompt_path is often a remote absolute
    # path (/home/geney/...). Rebuild the local path from source_run_dir + prompts/.
    source_run_dir = str(rec.get("_source_run_dir", "")).strip()
    if source_run_dir:
        parts = direct.parts
        if "prompts" in parts:
            idx = parts.index("prompts")
            rel = Path(*parts[idx:])
            candidate = Path(source_run_dir) / rel
            if candidate.exists():
                return candidate

        # Conservative fallback by basename if layout changed but file names match.
        by_name = list(Path(source_run_dir).glob(f"prompts/**/{direct.name}"))
        if by_name:
            return by_name[0]

    return None


def _read_prompt_text(rec: Dict[str, Any]) -> str:
    resolved = _resolve_prompt_path(rec)
    if resolved is not None:
        return resolved.read_text(encoding="utf-8")
    # Fallback: use preview if prompt file not available locally.
    return str(rec.get("prompt_preview", ""))


def _write_combined_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def _write_markdown(path: Path, records: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# Combined Future Report (Unabridged)")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat()}`")
    lines.append(f"- Entries: `{len(records)}`")
    lines.append("")

    for i, rec in enumerate(records, 1):
        t = rec.get("t")
        run_id = rec.get("run_id", "")
        model = rec.get("model", "")
        status = rec.get("status", "")
        latency = rec.get("latency_s", "")
        source = rec.get("_source_run_dir", "")
        prompt_path = rec.get("prompt_path", "")
        prompt_text = _read_prompt_text(rec)
        response_text = str(rec.get("response_text", ""))
        error_text = str(rec.get("error", ""))

        lines.append(f"## {i}. t={t}, run_id={run_id}, model={model}")
        lines.append(f"- status: `{status}`")
        lines.append(f"- latency_s: `{latency}`")
        lines.append(f"- source_run_dir: `{source}`")
        if prompt_path:
            lines.append(f"- prompt_path: `{prompt_path}`")
        lines.append("")
        lines.append("### Prompt (Full)")
        lines.append("```text")
        lines.append(_safe_block(prompt_text))
        lines.append("```")
        lines.append("")
        lines.append("### Response (Full)")
        lines.append("```text")
        lines.append(_safe_block(response_text))
        lines.append("```")
        lines.append("")
        if error_text:
            lines.append("### Error")
            lines.append("```text")
            lines.append(_safe_block(error_text))
            lines.append("```")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _extract_map_line(raw_text_obs: str) -> str:
    for line in raw_text_obs.splitlines():
        if line.strip().startswith(MAP_INTERESTING_PREFIX):
            return line.strip()
    if _filter_text_obs is not None:
        try:
            filtered = _filter_text_obs(raw_text_obs, strict_map_validation=False)
        except TypeError:
            filtered = _filter_text_obs(raw_text_obs)
        except Exception:
            filtered = ""
        if filtered:
            for line in filtered.splitlines():
                if line.strip().startswith(MAP_INTERESTING_PREFIX):
                    return line.strip()
    return ""


def _parse_map_entries_from_line(map_line: str) -> List[str]:
    if not map_line.startswith(MAP_INTERESTING_PREFIX):
        return []
    payload = map_line[len(MAP_INTERESTING_PREFIX):].strip()
    if not payload:
        return []
    starts = list(_MAP_COORD_PREFIX_RE.finditer(payload))
    if not starts:
        return []
    entries: List[str] = []
    for idx, match in enumerate(starts):
        start = match.start()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(payload)
        token = payload[start:end].strip().rstrip(",").strip()
        if token:
            entries.append(token)
    return entries


def _parse_map_entry_token(token: str) -> Optional[Tuple[int, int, str]]:
    m = _MAP_ENTRY_RE.match(token.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3).strip()


def _build_compact_state_from_raw(rec: Dict[str, Any], raw_text_obs: str, map_line: str) -> str:
    fields: Dict[str, str] = {}
    for line in raw_text_obs.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        fields[key.strip()] = val.strip()

    health = _safe_float(fields.get("Health"))
    food = _safe_float(fields.get("Food"))
    drink = _safe_float(fields.get("Drink"))
    energy = _safe_float(fields.get("Energy"))
    mana = _safe_float(fields.get("Mana"))
    xp = _safe_float(fields.get("XP"))
    direction = fields.get("Direction", "NA")
    floor = _safe_int(fields.get("Floor"))
    ladder_open = _safe_bool(fields.get("Ladder Open"))
    action_name = rec.get("action_name")
    action_id = rec.get("action_id")
    reward = _safe_float(rec.get("reward"))
    done = _safe_bool(rec.get("done"))

    stats_parts = [
        f"Health={health if health is not None else 'NA'}",
        f"Food={food if food is not None else 'NA'}",
        f"Drink={drink if drink is not None else 'NA'}",
        f"Energy={energy if energy is not None else 'NA'}",
        f"Mana={mana if mana is not None else 'NA'}",
        f"XP={xp if xp is not None else 'NA'}",
    ]
    action_val = action_name if action_name else action_id if action_id is not None else "NA"

    return "\n".join(
        [
            map_line if map_line else "Map (interesting tiles only): <unavailable>",
            "Stats: " + ", ".join(stats_parts),
            f"Direction={direction}, Floor={floor if floor is not None else 'NA'}, "
            f"LadderOpen={ladder_open if ladder_open is not None else 'NA'}",
            f"Action@t={action_val}, Reward@t={reward if reward is not None else 'NA'}, "
            f"Done@t={done if done is not None else 'NA'}",
        ]
    )


def _build_viewer_payload(trajectory_dir: Path) -> Dict[str, Any]:
    text_obs_path = trajectory_dir / "text_obs.jsonl"
    if not text_obs_path.exists():
        raise FileNotFoundError(f"Missing text_obs.jsonl for viewer: {text_obs_path}")

    states: List[Dict[str, Any]] = []
    min_row: Optional[int] = None
    max_row: Optional[int] = None
    min_col: Optional[int] = None
    max_col: Optional[int] = None

    with text_obs_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = _safe_int(rec.get("t"))
            if t is None:
                continue
            raw_text_obs = str(rec.get("raw_text_obs", ""))
            map_line = _extract_map_line(raw_text_obs)
            map_tokens = _parse_map_entries_from_line(map_line) if map_line else []

            parsed_entries: List[Dict[str, Any]] = []
            for token in map_tokens:
                parsed = _parse_map_entry_token(token)
                if parsed is None:
                    continue
                row, col, tile = parsed
                parsed_entries.append({"row": row, "col": col, "tile": tile})
                min_row = row if min_row is None else min(min_row, row)
                max_row = row if max_row is None else max(max_row, row)
                min_col = col if min_col is None else min(min_col, col)
                max_col = col if max_col is None else max(max_col, col)

            compact_state = _build_compact_state_from_raw(rec, raw_text_obs, map_line)
            states.append(
                {
                    "t": t,
                    "episode_id": _safe_int(rec.get("episode_id")),
                    "action_name": rec.get("action_name"),
                    "action_id": _safe_int(rec.get("action_id")),
                    "reward": _safe_float(rec.get("reward")),
                    "done": _safe_bool(rec.get("done")),
                    "state_text": compact_state,
                    "map_entries": parsed_entries,
                }
            )

    states.sort(key=lambda x: int(x["t"]))

    if min_row is None or max_row is None or min_col is None or max_col is None:
        min_row, max_row, min_col, max_col = -5, 5, -4, 4

    return {
        "bounds": {
            "min_row": int(min_row),
            "max_row": int(max_row),
            "min_col": int(min_col),
            "max_col": int(max_col),
        },
        "states": states,
    }


def _attach_viewer_frame_paths(
    viewer_payload: Dict[str, Any],
    *,
    frame_dir: Optional[Path],
    report_dir: Path,
) -> Dict[str, Any]:
    if frame_dir is None:
        return viewer_payload
    if not frame_dir.exists():
        return viewer_payload

    states = viewer_payload.get("states")
    if not isinstance(states, list):
        return viewer_payload

    for state in states:
        t = _safe_int(state.get("t"))
        if t is None:
            state["frame_path"] = ""
            continue
        frame_path = frame_dir / f"t_{t:05d}.png"
        if frame_path.exists():
            rel = os.path.relpath(frame_path, report_dir)
            state["frame_path"] = f"{rel}?v={int(frame_path.stat().st_mtime_ns)}"
        else:
            state["frame_path"] = ""
    return viewer_payload


def _infer_trajectory_dir(records: List[Dict[str, Any]]) -> Optional[Path]:
    trajectory_id = ""
    for rec in records:
        maybe = str(rec.get("trajectory_id", "")).strip()
        if maybe:
            trajectory_id = maybe
            break
    if not trajectory_id:
        return None

    cwd_candidate = Path.cwd() / "play_data" / "trajectory_records" / trajectory_id
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    repo_candidate = Path(__file__).resolve().parents[1] / "play_data" / "trajectory_records" / trajectory_id
    if repo_candidate.exists():
        return repo_candidate.resolve()
    return None


def _write_html(
    path: Path,
    records: List[Dict[str, Any]],
    *,
    collapse_prompts: bool,
    viewer_payload: Optional[Dict[str, Any]],
) -> None:
    html: List[str] = []
    inspection_timesteps: List[int] = sorted(
        {int(t) for t in [_safe_int(rec.get("t")) for rec in records] if t is not None}
    )
    viewer_json = _html_escape(json.dumps(viewer_payload, ensure_ascii=True)) if viewer_payload else ""
    inspection_json = _html_escape(json.dumps(inspection_timesteps, ensure_ascii=True))

    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html.append("<title>Combined Future Report (Unabridged)</title>")
    html.append(
        "<style>"
        "body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:18px;line-height:1.35;}"
        "pre{white-space:pre-wrap;word-break:break-word;background:#fafafa;border:1px solid #eee;padding:10px;}"
        "details{margin:10px 0;border:1px solid #ddd;padding:8px;border-radius:6px;}"
        ".meta{color:#444;font-size:13px;}"
        ".err{color:#a00;}"
        ".viewer-wrap{border:1px solid #ddd;border-radius:8px;padding:10px;margin:12px 0 18px 0;}"
        ".viewer-controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;}"
        ".viewer-controls input[type=range]{min-width:320px;flex:1 1 440px;}"
        ".viewer-frame{display:block;max-width:100%;height:auto;border:1px solid #ddd;background:#111;"
        "image-rendering:pixelated;image-rendering:crisp-edges;margin:8px 0;}"
        ".viewer-grid{display:grid;gap:2px;justify-content:start;margin:8px 0;border:1px solid #ddd;padding:8px;background:#f8f8f8;}"
        ".map-cell{width:28px;height:28px;display:flex;align-items:center;justify-content:center;border:1px solid #e6e6e6;"
        "font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#fff;}"
        ".map-cell.player{background:#d8f5d1;font-weight:700;}"
        ".map-cell.enemy{background:#ffe0e0;}"
        ".map-cell.water{background:#deefff;}"
        ".map-cell.path{background:#f7f0dd;}"
        ".map-cell.resource{background:#e6f7e6;}"
        ".map-cell.wall{background:#efefef;}"
        ".map-cell.darkness{background:#2f3340;color:#e8edf5;}"
        ".viewer-metrics{display:flex;gap:10px;flex-wrap:wrap;font-size:13px;margin-bottom:6px;}"
        "</style>"
    )
    html.append("</head><body>")
    html.append("<h1>Combined Future Report (Unabridged)</h1>")
    html.append(f"<p class='meta'>Generated: {datetime.now().isoformat()} | Entries: {len(records)}</p>")

    if viewer_payload and viewer_payload.get("states"):
        html.append("<h2>Trajectory Viewer</h2>")
        html.append(
            "<p class='meta'>Scrub the timeline to inspect observed game states and compare against model summaries.</p>"
        )
        html.append("<div class='viewer-wrap'>")
        html.append("<div class='viewer-controls'>")
        html.append("<button type='button' id='viewer-prev'>&lt;</button>")
        html.append("<input id='viewer-slider' type='range' min='0' max='0' step='1' value='0' />")
        html.append("<button type='button' id='viewer-next'>&gt;</button>")
        html.append("<label>t=<strong id='viewer-t'>0</strong></label>")
        html.append("<label>idx=<span id='viewer-idx'>0</span></label>")
        html.append("</div>")
        html.append("<div class='viewer-controls'>")
        html.append("<label for='viewer-jump'>Jump to inspected t:</label>")
        html.append("<select id='viewer-jump'><option value=''>-- choose --</option></select>")
        html.append("</div>")
        html.append("<div class='viewer-metrics' id='viewer-metrics'></div>")
        html.append("<img id='viewer-frame' class='viewer-frame' alt='Craftax frame' />")
        html.append("<div id='viewer-grid' class='viewer-grid'></div>")
        html.append("<pre id='viewer-state-text'></pre>")
        html.append("</div>")
        html.append(f"<script id='viewer-data' type='application/json'>{viewer_json}</script>")
        html.append(
            f"<script id='viewer-inspection-ts' type='application/json'>{inspection_json}</script>"
        )
        html.append(
            "<script>"
            "(function(){"
            "const dataEl=document.getElementById('viewer-data');"
            "const inspEl=document.getElementById('viewer-inspection-ts');"
            "if(!dataEl){return;}"
            "let payload={states:[],bounds:{min_row:-5,max_row:5,min_col:-4,max_col:4}};"
            "let inspectionTs=[];"
            "try{payload=JSON.parse(dataEl.textContent||'{}')||payload;}catch(_){payload=payload;}"
            "try{inspectionTs=JSON.parse(inspEl.textContent||'[]')||[];}catch(_){inspectionTs=[];}"
            "const states=Array.isArray(payload.states)?payload.states:[];"
            "const b=payload.bounds||{};"
            "const bounds={"
            "min_row:Number.isFinite(b.min_row)?b.min_row:-5,"
            "max_row:Number.isFinite(b.max_row)?b.max_row:5,"
            "min_col:Number.isFinite(b.min_col)?b.min_col:-4,"
            "max_col:Number.isFinite(b.max_col)?b.max_col:4"
            "};"
            "const slider=document.getElementById('viewer-slider');"
            "const prevBtn=document.getElementById('viewer-prev');"
            "const nextBtn=document.getElementById('viewer-next');"
            "const tEl=document.getElementById('viewer-t');"
            "const idxEl=document.getElementById('viewer-idx');"
            "const frameEl=document.getElementById('viewer-frame');"
            "const gridEl=document.getElementById('viewer-grid');"
            "const textEl=document.getElementById('viewer-state-text');"
            "const metricsEl=document.getElementById('viewer-metrics');"
            "const jumpEl=document.getElementById('viewer-jump');"
            "if(!slider||!gridEl||!textEl||states.length===0){"
            "if(textEl){textEl.textContent='No trajectory states available for viewer.';}return;}"
            "const tToIdx=new Map();"
            "for(let i=0;i<states.length;i+=1){tToIdx.set(states[i].t,i);}"
            "for(const t of inspectionTs){if(!tToIdx.has(t)){continue;}const opt=document.createElement('option');opt.value=String(t);opt.textContent='t='+String(t);jumpEl.appendChild(opt);}"
            "const tileSymbol=(tile)=>{"
            "const s=(tile||'').toLowerCase();"
            "if(!s){return ' ';}"
            "if(s.includes('player')){return 'P';}"
            "if(s.includes('ladder')){return 'L';}"
            "if(s.includes('path')){return '.';}"
            "if(s.includes('water')){return '~';}"
            "if(s.includes('tree')){return 'T';}"
            "if(s.includes('stone')||s.includes('ore')){return 'O';}"
            "if(s.includes('wall')){return '#';}"
            "if(s.includes('darkness')){return 'D';}"
            "if(s.includes('cow')||s.includes('skeleton')||s.includes('zombie')||s.includes('gnome')||s.includes('orc')||s.includes('troll')){return 'M';}"
            "return (tile||'?').charAt(0).toUpperCase();"
            "};"
            "const tileClass=(tile)=>{"
            "const s=(tile||'').toLowerCase();"
            "if(s.includes('player')){return 'player';}"
            "if(s.includes('cow')||s.includes('skeleton')||s.includes('zombie')||s.includes('gnome')||s.includes('orc')||s.includes('troll')){return 'enemy';}"
            "if(s.includes('water')){return 'water';}"
            "if(s.includes('path')){return 'path';}"
            "if(s.includes('tree')||s.includes('stone')||s.includes('ore')||s.includes('chest')||s.includes('gem')){return 'resource';}"
            "if(s.includes('wall')){return 'wall';}"
            "if(s.includes('darkness')){return 'darkness';}"
            "return '';"
            "};"
            "const esc=(x)=>String(x).replace(/[&<>\"']/g,(m)=>({ '&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;' }[m]));"
            "const cols=bounds.max_col-bounds.min_col+1;"
            "gridEl.style.gridTemplateColumns='repeat('+String(cols)+', 28px)';"
            "const render=(idx)=>{"
            "const s=states[idx]||states[0];"
            "slider.value=String(idx);"
            "idxEl.textContent=String(idx);"
            "tEl.textContent=String(s.t);"
            "const fp=String(s.frame_path||'');"
            "if(frameEl&&fp){"
            "frameEl.style.display='block';"
            "frameEl.setAttribute('src', encodeURI(fp));"
            "gridEl.style.display='none';"
            "}else{"
            "if(frameEl){frameEl.style.display='none';frameEl.removeAttribute('src');}"
            "gridEl.style.display='grid';"
            "const byCoord=new Map();"
            "for(const e of (s.map_entries||[])){byCoord.set(String(e.row)+','+String(e.col),e.tile||'');}"
            "let cells='';"
            "for(let r=bounds.min_row;r<=bounds.max_row;r+=1){"
            "for(let c=bounds.min_col;c<=bounds.max_col;c+=1){"
            "const key=String(r)+','+String(c);"
            "const tile=byCoord.get(key)||'';"
            "const sym=tileSymbol(tile);"
            "const cls=tileClass(tile);"
            "const title=tile?key+': '+tile:key+': (empty / not listed)';"
            "cells+='<div class=\"map-cell '+cls+'\" title=\"'+esc(title)+'\">'+esc(sym)+'</div>';"
            "}"
            "}"
            "gridEl.innerHTML=cells;"
            "}"
            "const metrics=["
            "'episode='+(s.episode_id===null||s.episode_id===undefined?'NA':String(s.episode_id)),"
            "'action='+(s.action_name||String(s.action_id||'NA')),"
            "'reward='+(s.reward===null||s.reward===undefined?'NA':String(s.reward)),"
            "'done='+(s.done===null||s.done===undefined?'NA':String(s.done))"
            "];"
            "metricsEl.textContent='';"
            "for(const m of metrics){const span=document.createElement('span');span.textContent=m;metricsEl.appendChild(span);}"
            "textEl.textContent=String(s.state_text||'');"
            "};"
            "slider.min='0';"
            "slider.max=String(states.length-1);"
            "slider.step='1';"
            "slider.addEventListener('input',()=>render(Number(slider.value||'0')));"
            "if(prevBtn){prevBtn.addEventListener('click',()=>{const n=Math.max(0,Number(slider.value||'0')-1);render(n);});}"
            "if(nextBtn){nextBtn.addEventListener('click',()=>{const n=Math.min(states.length-1,Number(slider.value||'0')+1);render(n);});}"
            "if(jumpEl){jumpEl.addEventListener('change',()=>{const t=Number(jumpEl.value);if(!Number.isFinite(t)||!tToIdx.has(t)){return;}render(tToIdx.get(t));});}"
            "render(0);"
            "})();"
            "</script>"
        )
    else:
        html.append("<p class='meta'>Trajectory viewer unavailable (no trajectory data found).</p>")

    for i, rec in enumerate(records, 1):
        t = rec.get("t")
        run_id = rec.get("run_id", "")
        model = rec.get("model", "")
        status = rec.get("status", "")
        latency = rec.get("latency_s", "")
        source = rec.get("_source_run_dir", "")
        prompt_path = rec.get("prompt_path", "")
        prompt_text = _read_prompt_text(rec)
        response_text = str(rec.get("response_text", ""))
        error_text = str(rec.get("error", ""))

        html.append("<details>")
        html.append(
            "<summary><strong>"
            f"{i}. t={_html_escape(str(t))}, run_id={_html_escape(str(run_id))}, model={_html_escape(str(model))}"
            "</strong></summary>"
        )
        html.append("<div class='meta'>")
        html.append(f"status={_html_escape(str(status))} | latency_s={_html_escape(str(latency))}<br>")
        html.append(f"source_run_dir={_html_escape(str(source))}<br>")
        if prompt_path:
            html.append(f"prompt_path={_html_escape(str(prompt_path))}")
        html.append("</div>")
        html.append("<details>")
        html.append("<summary><strong>Prompt (Full)</strong></summary>")
        html.append(f"<pre>{_html_escape(prompt_text)}</pre>")
        html.append("</details>")
        html.append("<details>")
        html.append("<summary><strong>Response (Full)</strong></summary>")
        html.append(f"<pre>{_html_escape(response_text)}</pre>")
        html.append("</details>")
        if error_text:
            html.append("<details>")
            html.append("<summary><strong class='err'>Error</strong></summary>")
            html.append(f"<pre class='err'>{_html_escape(error_text)}</pre>")
            html.append("</details>")
        html.append("</details>")

    html.append("</body></html>")
    path.write_text("\n".join(html), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    run_dirs = [Path(p).resolve() for p in args.run_dir]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestep_filter = _parse_timesteps(args.timesteps)
    records = _load_records(run_dirs)

    if not args.include_errors:
        records = [r for r in records if str(r.get("status", "")) == "ok"]
    if timestep_filter is not None:
        records = [r for r in records if _safe_int(r.get("t")) in timestep_filter]

    if args.dedupe_by_timestep:
        records = _dedupe_latest_by_timestep(records)
    else:
        records = sorted(records, key=_record_sort_key)

    viewer_payload: Optional[Dict[str, Any]] = None
    if not args.disable_viewer:
        trajectory_dir = args.trajectory_dir.resolve() if args.trajectory_dir else _infer_trajectory_dir(records)
        if trajectory_dir is not None and trajectory_dir.exists():
            viewer_payload = _build_viewer_payload(trajectory_dir)
            viewer_frame_dir: Optional[Path] = None
            if args.viewer_frame_dir is not None:
                cand = args.viewer_frame_dir.resolve()
                if cand.exists():
                    viewer_frame_dir = cand
            else:
                default_frame_dir = (trajectory_dir / "render_frames_bs16").resolve()
                if default_frame_dir.exists():
                    viewer_frame_dir = default_frame_dir
            viewer_payload = _attach_viewer_frame_paths(
                viewer_payload,
                frame_dir=viewer_frame_dir,
                report_dir=output_dir,
            )
            print(f"viewer_trajectory_dir: {trajectory_dir}")
            print(f"viewer_states: {len(viewer_payload.get('states', []))}")
            print(f"viewer_frame_dir: {viewer_frame_dir if viewer_frame_dir is not None else ''}")
        else:
            print("viewer_trajectory_dir: <not found>")

    combined_jsonl = output_dir / "combined_records.jsonl"
    md_path = output_dir / "combined_unabridged_report.md"
    html_path = output_dir / "combined_unabridged_report.html"

    _write_combined_jsonl(combined_jsonl, records)
    _write_markdown(md_path, records)
    _write_html(
        html_path,
        records,
        collapse_prompts=bool(args.collapse_prompts),
        viewer_payload=viewer_payload,
    )

    print(combined_jsonl)
    print(md_path)
    print(html_path)
    print(f"entries={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
