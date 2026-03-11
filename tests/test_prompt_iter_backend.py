import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import prompt_iter_backend as backend


_ENTRY_RE = re.compile(r"-?\d+\s*,\s*-?\d+\s*:[^,]+")
_COORD_PREFIX_RE = re.compile(r"-?\d+\s*,\s*-?\d+\s*:")


def _assert_payload_tokens_valid(payload: str) -> None:
    starts = list(_COORD_PREFIX_RE.finditer(payload))
    tokens = []
    for i, match in enumerate(starts):
        start = match.start()
        end = starts[i + 1].start() if i + 1 < len(starts) else len(payload)
        token = payload[start:end].strip().rstrip(",").strip()
        if token:
            tokens.append(token)
    assert tokens, "Expected at least one coordinate token"
    for token in tokens:
        assert _ENTRY_RE.fullmatch(token), f"Malformed token: {token!r}"


def test_fixed_states_manifest_loads_expected_set() -> None:
    states = backend.load_fixed_states(backend.DEFAULT_MANIFEST)
    assert len(states) == 10

    state_ids = {state.state_id for state in states}
    expected = {
        "golden_step_0006",
        "golden_step_0240",
        "golden_step_0619",
        "golden_step_1103",
        "golden_step_1456",
        "traj_20260310_134702_t0000",
        "traj_20260310_134702_t0155",
        "traj_20260310_134702_t0240",
        "traj_20260310_134702_t0425",
        "traj_20260310_134702_t0567",
    }
    assert state_ids == expected

    trajectory_t = {
        state.t
        for state in states
        if state.source_kind == "trajectory_jsonl"
    }
    assert trajectory_t == {0, 155, 240, 425, 567}


@pytest.mark.parametrize("state_id", [
    "golden_step_0006",
    "golden_step_1103",
    "traj_20260310_134702_t0155",
    "traj_20260310_134702_t0425",
])
def test_loaded_state_map_lines_have_integrity(state_id: str) -> None:
    states = backend.states_by_id(backend.load_fixed_states(backend.DEFAULT_MANIFEST))
    state = states[state_id]
    map_line = state.map_line()
    assert map_line.startswith(backend.MAP_INTERESTING_PREFIX)
    payload = map_line[len(backend.MAP_INTERESTING_PREFIX):]
    _assert_payload_tokens_valid(payload)


def test_build_prompt_uses_generation_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"SYS={messages[0]['content']}\\nUSR={messages[1]['content']}\\n<GEN>"

    monkeypatch.setattr(backend, "_load_tokenizer", lambda _model_id: DummyTokenizer())

    sections = backend.default_prompt_sections("future_based_opt")
    prompt = backend.build_prompt(
        "Map (interesting tiles only): 0, 1:tree\nInventory:\nHealth: 9.0\n",
        sections,
        model_id="dummy",
    )

    assert prompt.endswith(sections.generation_prefix)
    assert "YOUR CURRENT GAME STATE" in prompt
    assert "Map (interesting tiles only): 0, 1:tree" in prompt


def test_run_state_falls_back_to_chat_when_transformers_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = backend.FixedState(
        state_id="dummy",
        label="Dummy",
        source_kind="unit",
        source_path="dummy",
        tags=[],
        raw_text_obs="Map: 0,0:tree\nInventory:\nHealth: 9.0\n",
        filtered_text_obs="Map (interesting tiles only): 0, 0:tree\nInventory:\nHealth: 9.0\n",
    )
    sections = backend.default_prompt_sections("default")

    def _raise_missing(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'transformers'")

    monkeypatch.setattr(backend, "build_prompt", _raise_missing)
    monkeypatch.setattr(
        backend,
        "run_chat_completion",
        lambda *_args, **_kwargs: {"choices": [{"message": {"content": "fallback ok"}}]},
    )

    result = backend.run_state(state, sections, prefer_chat_completions=False)
    assert result["request_mode"] == "chat_completions_fallback"
    assert result["response_text"] == "fallback ok"
    assert "_prompt_iter_note" in result["response_json"]


def test_run_state_prefers_chat_completions_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = backend.FixedState(
        state_id="dummy",
        label="Dummy",
        source_kind="unit",
        source_path="dummy",
        tags=[],
        raw_text_obs="Map: 0,0:tree\nInventory:\nHealth: 9.0\n",
        filtered_text_obs="Map (interesting tiles only): 0, 0:tree\nInventory:\nHealth: 9.0\n",
    )
    sections = backend.default_prompt_sections("default")

    def _should_not_call_build_prompt(*_args, **_kwargs):
        raise AssertionError("build_prompt should not be used in prefer_chat_completions mode")

    monkeypatch.setattr(backend, "build_prompt", _should_not_call_build_prompt)
    monkeypatch.setattr(
        backend,
        "run_chat_completion",
        lambda *_args, **_kwargs: {"choices": [{"message": {"content": "chat mode ok"}}]},
    )

    result = backend.run_state(state, sections, prefer_chat_completions=True)
    assert result["request_mode"] == "chat_completions"
    assert result["response_text"] == "chat mode ok"
