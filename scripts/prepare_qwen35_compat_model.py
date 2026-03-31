#!/usr/bin/env python3
"""
Prepare a local compatibility copy of a Qwen3.5 model config for vLLM runs.

This creates a lightweight mirror directory with symlinks to the HF snapshot files,
then patches `config.json` with one of two modes:

1) preserve-native (default):
   - keep qwen3_5 architecture/model_type as-is
   - copy key `text_config` fields to top-level if missing (for spec-parser compatibility)
   - keep `text_config` present

2) force-qwen3 (`--force-qwen3`):
   - rewrite model_type -> qwen3
   - rewrite architectures -> [Qwen3ForCausalLM]
   - flatten `text_config` to top-level and remove nested `text_config`
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def _link_tree(src_root: Path, dst_root: Path) -> None:
    for src in src_root.rglob("*"):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)


def _resolve_snapshot_root(model_id_or_path: str) -> Path:
    local_path = Path(model_id_or_path).expanduser()
    if local_path.exists():
        return local_path.resolve()
    return Path(snapshot_download(repo_id=model_id_or_path))


def _patch_config(
    cfg_path: Path,
    force_qwen3: bool,
    target_architecture: str | None,
) -> dict:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict):
        flat_text = dict(text_cfg)
        if (
            flat_text.get("max_window_layers") is None
            and flat_text.get("num_hidden_layers") is not None
        ):
            flat_text["max_window_layers"] = int(flat_text["num_hidden_layers"])

        # Copy text config fields up for components that expect top-level values.
        for k, v in flat_text.items():
            if k not in cfg or cfg.get(k) is None:
                cfg[k] = v
        cfg["max_window_layers"] = flat_text.get(
            "max_window_layers", cfg.get("max_window_layers")
        )

        if force_qwen3:
            flat_text["model_type"] = "qwen3"
            cfg.pop("text_config", None)

    if force_qwen3:
        cfg["model_type"] = "qwen3"
        cfg["architectures"] = ["Qwen3ForCausalLM"]
    else:
        # Force vLLM/transformers built-in config resolution for qwen3_5.
        # Remote AutoConfig mappings can return a plain dict text_config, which
        # fails speculative config validation expecting attribute access.
        cfg.pop("auto_map", None)
        if target_architecture:
            cfg["architectures"] = [target_architecture]

    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg


def _sample_weight_keys(snapshot_dir: Path, limit: int = 40) -> tuple[int | None, list[str]]:
    idx_path = snapshot_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        return None, []
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return None, []
    wm = idx.get("weight_map")
    if not isinstance(wm, dict):
        return None, []
    keys = list(wm.keys())
    return len(keys), keys[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="HF model id")
    parser.add_argument("--output-dir", required=True, help="Local compat model dir")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing output dir before recreating",
    )
    parser.add_argument(
        "--force-qwen3",
        action="store_true",
        help="Rewrite config to legacy qwen3 architecture mode (v0.14 compatibility).",
    )
    parser.add_argument(
        "--target-architecture",
        default=None,
        help=(
            "Optional single architecture override for preserved qwen3_5 configs "
            "(e.g. Qwen3_5ForCausalLM)."
        ),
    )
    args = parser.parse_args()

    snapshot = _resolve_snapshot_root(args.model_id)
    out_dir = Path(args.output_dir)
    if out_dir.exists() and not args.no_clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _link_tree(snapshot, out_dir)
    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in snapshot mirror: {cfg_path}")

    # Replace symlinked config with a real patched file.
    if cfg_path.is_symlink():
        cfg_path.unlink()
        shutil.copy2(snapshot / "config.json", cfg_path)

    cfg = _patch_config(
        cfg_path,
        force_qwen3=args.force_qwen3,
        target_architecture=args.target_architecture,
    )
    text_cfg = cfg if isinstance(cfg, dict) else {}
    key_count, key_samples = _sample_weight_keys(snapshot, limit=40)

    print(f"model_id={args.model_id}")
    print(f"snapshot={snapshot}")
    print(f"compat_dir={out_dir}")
    print(f"compat_mode={'force-qwen3' if args.force_qwen3 else 'preserve-native'}")
    print(f"target_architecture={args.target_architecture}")
    print(f"patched_model_type={cfg.get('model_type')}")
    print(f"patched_architectures={cfg.get('architectures')}")
    print(f"flat_model_type={text_cfg.get('model_type')}")
    print(f"flat_max_window_layers={text_cfg.get('max_window_layers')}")
    print(f"flat_num_hidden_layers={text_cfg.get('num_hidden_layers')}")
    print(f"flat_num_attention_heads={text_cfg.get('num_attention_heads')}")
    print(f"snapshot_weight_key_count={key_count}")
    for i, key in enumerate(key_samples):
        print(f"snapshot_weight_key_sample_{i:02d}={key}")


if __name__ == "__main__":
    main()
