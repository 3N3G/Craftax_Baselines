import re
from pathlib import Path

import numpy as np
import pytest

from labelling.obs_to_text import obs_to_text
from utils.llm_prompts import (
    MAP_INTERESTING_PREFIX,
    ensure_valid_interesting_map,
    filter_text_obs,
)


_ENTRY_RE = re.compile(r"-?\d+\s*,\s*-?\d+\s*:[^,]+")
_COORD_PREFIX_RE = re.compile(r"-?\d+\s*,\s*-?\d+\s*:")


def _assert_payload_has_valid_row_col_tokens(payload: str) -> None:
    starts = list(_COORD_PREFIX_RE.finditer(payload))
    tokens = []
    for i, match in enumerate(starts):
        start = match.start()
        end = starts[i + 1].start() if i + 1 < len(starts) else len(payload)
        token = payload[start:end].strip().rstrip(",").strip()
        if token:
            tokens.append(token)
    assert tokens, "Expected at least one map token"
    for token in tokens:
        assert _ENTRY_RE.fullmatch(token), f"Malformed token: {token!r}"


def _interesting_payload(text: str) -> str:
    for line in text.splitlines():
        if line.startswith(MAP_INTERESTING_PREFIX):
            return line[len(MAP_INTERESTING_PREFIX):]
    raise AssertionError("Missing 'Map (interesting tiles only)' line")


def test_compact_map_regression_keeps_row_col_pairs() -> None:
    raw = (
        "Map: -5,-4:water, -4,-4:water, 3,-1:tree\n"
        "Inventory:\n"
        "Health: 9.0\n"
    )
    filtered = filter_text_obs(raw, strict_map_validation=True)
    payload = _interesting_payload(filtered)
    assert "-5, -4:water" in payload
    assert "-4, -4:water" in payload
    assert "3, -1:tree" in payload
    assert "-5:water" not in payload
    _assert_payload_has_valid_row_col_tokens(payload)


def test_multiline_map_parsing_filters_background() -> None:
    raw = (
        "Map:\n"
        "-5, -4: grass\n"
        "-4, -4: tree\n"
        "-3, -4: Cow on grass\n"
        "\n"
        "Inventory:\n"
        "Health: 9.0\n"
    )
    filtered = filter_text_obs(raw, strict_map_validation=True)
    payload = _interesting_payload(filtered)
    assert "-4, -4:tree" in payload
    assert "-3, -4:Cow on grass" in payload
    assert "grass" not in payload or "Cow on grass" in payload
    _assert_payload_has_valid_row_col_tokens(payload)


def test_entity_on_background_tile_is_retained() -> None:
    raw = (
        "Map: 0,0:Cow on grass, 0,1:grass, 0,2:sand\n"
        "Inventory:\n"
        "Health: 9.0\n"
    )
    filtered = filter_text_obs(raw, strict_map_validation=True)
    payload = _interesting_payload(filtered)
    assert "0, 0:Cow on grass" in payload
    assert "0, 1:grass" not in payload
    assert "0, 2:sand" not in payload


def test_runtime_guard_rejects_malformed_interesting_map_line() -> None:
    malformed = (
        "Map (interesting tiles only): -5:tree, -4:stone\n"
        "Inventory:\n"
        "Health: 9.0\n"
    )
    with pytest.raises(ValueError):
        ensure_valid_interesting_map(malformed)


def test_integration_obs_to_text_output_produces_valid_coordinates() -> None:
    obs_path = Path(
        "golden_examples/game_20260225_211753/bundles/step_0006/obs_before.npy"
    )
    obs = np.load(obs_path)
    raw = obs_to_text(obs)
    filtered = filter_text_obs(raw, strict_map_validation=True)
    payload = _interesting_payload(filtered)
    entries = _ENTRY_RE.findall(payload)
    assert entries, "Expected at least one interesting map token"
    _assert_payload_has_valid_row_col_tokens(payload)
