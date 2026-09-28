"""Prompt loading and request rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from factory import message, tool_call, tool_result, trace

from agent_claimcheck.judge.render import (
    JudgeSpec,
    Prompt,
    PromptError,
    load_prompt,
    openrouter_extra_body,
    render_body,
    render_claims,
    render_request,
    render_steps,
    request_sha256,
)
from agent_claimcheck.redact import detector_view
from agent_claimcheck.schema import Trace

CANARY = "canary-9a41f0e2-do-not-leak"


def test_load_prompt_builtin_claim_audit() -> None:
    prompt = load_prompt("claim-audit")
    assert prompt.name == "claim-audit"
    assert prompt.version == 1
    assert "$instruction" in prompt.text
    assert "$steps" in prompt.text
    assert "$claims" in prompt.text
    assert len(prompt.sha256) == 64


def test_load_prompt_builtin_claim_by_claim() -> None:
    prompt = load_prompt("claim-by-claim")
    assert prompt.name == "claim-by-claim"
    assert prompt.version == 1
    assert '"claims"' in prompt.text


def test_load_prompt_missing_front_matter(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("no front matter here $instruction", encoding="utf-8")
    with pytest.raises(PromptError):
        load_prompt(path)


def test_load_prompt_missing_name(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("---\nversion: 1\n---\nbody $instruction\n", encoding="utf-8")
    with pytest.raises(PromptError):
        load_prompt(path)


def test_render_body_missing_placeholder_raises_prompt_error() -> None:
    prompt = Prompt(name="x", version=1, text="hello $unknown", sha256="x")
    with pytest.raises(PromptError, match="unknown"):
        render_body(prompt, instruction="i", steps="s", claims="c")


def test_render_body_literal_dollar() -> None:
    prompt = Prompt(name="x", version=1, text="price is $$5, task: $instruction", sha256="x")
    rendered = render_body(prompt, instruction="book it", steps="", claims="")
    assert rendered == "price is $5, task: book it"


def test_render_steps_includes_header_and_fields() -> None:
    steps = [
        tool_call(0, "calendar.create_event", {"title": "sync"}),
        tool_result(
            1, "calendar.create_event", ok=True, output={"event_id": "e1", "status": "confirmed"}
        ),
    ]
    rendered = render_steps(steps)
    assert "[0] tool_call agent calendar.create_event" in rendered
    assert "[1] tool_result tool calendar.create_event" in rendered
    assert "ok: true" in rendered
    assert "event_id" in rendered and "e1" in rendered


def test_render_steps_truncates_long_output_field() -> None:
    long_value = "x" * 3000
    steps = [tool_result(0, "fs.read_file", output={"content": long_value, "path": "a.py"})]
    rendered = render_steps(steps)
    assert "…[truncated 1002 chars]" in rendered
    # The other field is untouched by the long field's truncation.
    assert '"path":"a.py"' in rendered or "a.py" in rendered


def test_render_claims_includes_final_message_and_claims() -> None:
    t = trace("t1", "booking", [message(0, "user", "book it")], text="All booked.")
    view = detector_view(t)
    rendered = render_claims(view.final_claim)
    assert "All booked." in rendered
    assert "resolved claims" in rendered


def test_render_request_shape_and_json_mode() -> None:
    t = trace("t1", "booking", [message(0, "user", "book it")], text="Done.")
    view = detector_view(t)
    prompt = load_prompt("claim-audit")
    spec = JudgeSpec(model="test/model", temperature=0.0, max_tokens=400, json_mode=True)
    body = render_request(view, prompt, spec)
    assert body["model"] == "test/model"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 400
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [{"role": "user", "content": body["messages"][0]["content"]}]
    assert len(body["messages"]) == 1


def test_render_request_no_json_mode_omits_response_format() -> None:
    t = trace("t1", "booking", [message(0, "user", "book it")], text="Done.")
    view = detector_view(t)
    prompt = load_prompt("claim-audit")
    spec = JudgeSpec(model="m", json_mode=False)
    body = render_request(view, prompt, spec)
    assert "response_format" not in body


def test_render_request_extra_body_merged() -> None:
    t = trace("t1", "booking", [message(0, "user", "book it")], text="Done.")
    view = detector_view(t)
    prompt = load_prompt("claim-audit")
    spec = JudgeSpec(model="m", extra_body={"provider": {"require_parameters": True}})
    body = render_request(view, prompt, spec)
    assert body["provider"] == {"require_parameters": True}


def test_render_request_deterministic() -> None:
    t = trace(
        "t1",
        "booking",
        [
            tool_call(0, "calendar.create_event", {"title": "sync"}),
            tool_result(1, "calendar.create_event", ok=True, output={"event_id": "e1"}),
        ],
        text="Booked.",
        claims=[("booked", {"start": "2026-01-01T10:00:00+00:00"})],
    )
    view = detector_view(t)
    prompt = load_prompt("claim-audit")
    spec = JudgeSpec(model="m")
    body1 = render_request(view, prompt, spec)
    body2 = render_request(view, prompt, spec)
    assert body1 == body2
    assert request_sha256(body1) == request_sha256(body2)


def test_request_sha256_stable_across_key_order() -> None:
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert request_sha256(a) == request_sha256(b)


@pytest.mark.parametrize("prompt_name", ["claim-audit", "claim-by-claim"])
def test_canary_never_reaches_a_rendered_prompt_or_request_body(prompt_name: str) -> None:
    raw = {
        "schema": "agent-trace/v1",
        "trace_id": "t1",
        "source": "test/0.0.1",
        "task": {"id": "task-1", "domain": "booking", "instruction": "book a table"},
        "steps": [
            {
                "i": 0,
                "ts": "2026-01-01T00:00:00+00:00",
                "kind": "tool_call",
                "role": "agent",
                "name": "calendar.create_event",
            },
        ],
        "final_claim": {"text": "Booked.", "claims": [{"type": "booked", "subject": {}}]},
        "ground_truth": {
            "outcome": "success",
            "checked_by": "human",
            "details": {"note": CANARY},
        },
        "meta": {"note": CANARY},
    }
    t = Trace.model_validate(raw)
    view = detector_view(t)
    prompt = load_prompt(prompt_name)
    spec = JudgeSpec(model="m")
    body = render_request(view, prompt, spec)

    assert CANARY not in body["messages"][0]["content"]
    assert CANARY not in json.dumps(body)


def test_openrouter_extra_body_only_for_openrouter_host() -> None:
    assert openrouter_extra_body(
        "https://openrouter.ai/api/v1", json_mode=True, supports_reasoning=True
    ) == {
        "provider": {"require_parameters": True},
        "reasoning": {"enabled": False},
    }
    assert (
        openrouter_extra_body("https://example.com/v1", json_mode=True, supports_reasoning=True)
        == {}
    )


def test_openrouter_extra_body_no_reasoning_key_when_unsupported() -> None:
    extra = openrouter_extra_body(
        "https://openrouter.ai/api/v1", json_mode=True, supports_reasoning=False
    )
    assert "reasoning" not in extra
    assert extra == {"provider": {"require_parameters": True}}


def test_openrouter_extra_body_no_provider_when_not_json_mode() -> None:
    extra = openrouter_extra_body(
        "https://openrouter.ai/api/v1", json_mode=False, supports_reasoning=False
    )
    assert extra == {}
