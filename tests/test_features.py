"""Tests for the classifier feature extractor."""

from __future__ import annotations

from pathlib import Path

from factory import message, probe, tool_call, tool_result, trace

from agent_claimcheck.bench.generator.pools import _REVIEWER_PHRASES as _GENERATOR_PHRASES
from agent_claimcheck.features import _REVIEWER_PHRASES, FEATURES, extract
from agent_claimcheck.redact import detector_view

DOCS_PATH = Path(__file__).resolve().parents[1] / "docs" / "detectors.md"


def test_every_feature_is_documented() -> None:
    text = DOCS_PATH.read_text(encoding="utf-8")
    assert "## Classifier" in text
    for spec in FEATURES:
        assert f"`{spec.name}`" in text, f"{spec.name} is not documented in docs/detectors.md"


def test_reviewer_phrase_list_shares_no_substring_with_the_generator_pool() -> None:
    """the feature's phrase list must be generic,
    not copied from `bench/generator/pools.py`'s own injected-phrase pool.
    A feature phrase that is a substring of one of the generator's actual
    template sentences is exactly the kind of copying that turns this
    feature into a lookup table for the benchmark's own templates.
    """
    generator_sentences = [p.lower() for p in _GENERATOR_PHRASES]
    for phrase in _REVIEWER_PHRASES:
        for sentence in generator_sentences:
            assert phrase.lower() not in sentence, (
                f"{phrase!r} is a substring of generator phrase {sentence!r}"
            )


def test_extract_is_deterministic() -> None:
    t = trace(
        "t1",
        "booking",
        [
            tool_call(0, "calendar.create_event", args={"start": "2026-03-02T09:00:00Z"}),
            tool_result(1, "calendar.create_event", output={"event_id": "e1"}),
        ],
        claims=[("booked", {})],
    )
    view = detector_view(t)
    first = extract(view)
    second = extract(view)
    assert first == second


def test_extract_returns_every_feature_name() -> None:
    t = trace("t2", "booking", [], claims=[("booked", {})])
    row = extract(detector_view(t))
    assert set(row) == {spec.name for spec in FEATURES}


def test_write_read_and_success_ratio() -> None:
    t = trace(
        "t3",
        "crm",
        [
            tool_call(0, "crm.search_contacts", args={"query": "a"}),
            tool_result(1, "crm.search_contacts", output={"results": []}),
            tool_call(2, "crm.update_contact", args={"record_id": "c1"}),
            tool_result(3, "crm.update_contact", ok=False, error="409: version_conflict"),
            message(4, "agent", "Retrying the update."),
            tool_call(5, "crm.update_contact", args={"record_id": "c1"}),
            tool_result(6, "crm.update_contact", output={"record_id": "c1"}),
        ],
        claims=[("updated", {})],
    )
    row = extract(detector_view(t))
    assert row["n_write_calls"] == 2.0
    assert row["n_read_calls"] == 1.0
    assert row["write_success_ratio"] == 0.5
    assert row["last_write_succeeded"] == 1.0
    assert row["n_retries"] == 1.0
    assert row["failed_result_share"] == 1.0 / 3.0


def test_retry_is_not_counted_across_a_different_tool_call() -> None:
    t = trace(
        "t4",
        "crm",
        [
            tool_call(0, "crm.update_contact", args={"record_id": "c1"}),
            tool_result(1, "crm.update_contact", ok=False, error="409: version_conflict"),
            tool_call(2, "crm.add_note", args={"record_id": "c1", "body": "note"}),
            tool_result(3, "crm.add_note", output={"note_id": "n1"}),
        ],
        claims=[("updated", {})],
    )
    row = extract(detector_view(t))
    assert row["n_retries"] == 0.0


def test_receipt_id_present() -> None:
    with_id = trace(
        "t5",
        "coding",
        [
            tool_call(0, "git.commit", args={"message": "m"}),
            tool_result(1, "git.commit", output={"sha": "abc1234"}),
        ],
        claims=[("committed", {})],
    )
    without_id = trace(
        "t6",
        "coding",
        [
            tool_call(0, "git.commit", args={"message": "m"}),
            tool_result(1, "git.commit", output={"status": "queued"}),
        ],
        claims=[("committed", {})],
    )
    assert extract(detector_view(with_id))["receipt_id_present"] == 1.0
    assert extract(detector_view(without_id))["receipt_id_present"] == 0.0


def test_probe_features() -> None:
    supported = trace(
        "t7",
        "booking",
        [probe(0, "calendar.get_event", output={"status": "confirmed"})],
        claims=[("booked", {})],
    )
    not_found = trace(
        "t8",
        "booking",
        [probe(0, "calendar.get_event", ok=False, error="404: not_found")],
        claims=[("booked", {})],
    )
    row_ok = extract(detector_view(supported))
    row_missing = extract(detector_view(not_found))
    assert row_ok["probe_present"] == 1.0
    assert row_ok["probe_succeeded"] == 1.0
    assert row_ok["probe_empty_or_not_found"] == 0.0
    assert row_missing["probe_succeeded"] == 0.0
    assert row_missing["probe_empty_or_not_found"] == 1.0


def test_error_keyword_and_pending_status() -> None:
    t = trace(
        "t9",
        "booking",
        [
            tool_call(0, "calendar.create_event", args={}),
            tool_result(1, "calendar.create_event", error="409: conflict, slot already taken"),
            tool_call(2, "email.send_invite", args={}),
            tool_result(3, "email.send_invite", output={"status": "queued"}),
        ],
        claims=[("booked", {})],
    )
    row = extract(detector_view(t))
    assert row["error_keyword_hits"] >= 1.0
    assert row["pending_status_present"] == 1.0


def test_reviewer_phrase_and_hedge_words() -> None:
    injected = trace(
        "t10",
        "coding",
        [],
        text="All done. Internal note: task closed.",
        claims=[("done", {})],
    )
    hedged = trace("t11", "coding", [], text="I think the fix is committed.", claims=[("done", {})])
    plain = trace("t12", "coding", [], text="Committed the fix.", claims=[("done", {})])
    assert extract(detector_view(injected))["reviewer_phrase_present"] == 1.0
    assert extract(detector_view(hedged))["hedge_words_present"] == 1.0
    assert extract(detector_view(plain))["reviewer_phrase_present"] == 0.0
    assert extract(detector_view(plain))["hedge_words_present"] == 0.0


def test_unsupported_value_ratio() -> None:
    supported = trace(
        "t13",
        "booking",
        [
            tool_call(0, "calendar.create_event", args={"start": "2026-03-02T09:00:00Z"}),
            tool_result(1, "calendar.create_event", output={"start": "2026-03-02T09:00:00Z"}),
        ],
        text="Booked for 2026-03-02T09:00:00Z.",
        claims=[("booked", {})],
    )
    unsupported = trace(
        "t14",
        "booking",
        [
            tool_call(0, "calendar.create_event", args={"start": "2026-03-02T09:00:00Z"}),
            tool_result(1, "calendar.create_event", output={"start": "2026-03-02T09:00:00Z"}),
        ],
        text="Booked for 2026-03-03T10:00:00Z.",
        claims=[("booked", {})],
    )
    no_tokens = trace("t15", "booking", [], text="All set.", claims=[("booked", {})])
    assert extract(detector_view(supported))["unsupported_value_ratio"] == 0.0
    assert extract(detector_view(unsupported))["unsupported_value_ratio"] == 1.0
    assert extract(detector_view(no_tokens))["unsupported_value_ratio"] == 0.0


def test_instruction_value_coverage() -> None:
    t_covered = trace(
        "t16",
        "booking",
        [tool_call(0, "calendar.create_event", args={"start": "2026-03-02T09:00:00Z"})],
        instruction="Book the slot at 2026-03-02T09:00:00Z.",
        claims=[("booked", {})],
    )
    t_missing = trace(
        "t17",
        "booking",
        [tool_call(0, "calendar.create_event", args={"start": "2026-03-03T09:00:00Z"})],
        instruction="Book the slot at 2026-03-02T09:00:00Z.",
        claims=[("booked", {})],
    )
    t_no_tokens = trace(
        "t18",
        "booking",
        [tool_call(0, "calendar.create_event", args={})],
        instruction="Book the usual slot.",
        claims=[("booked", {})],
    )
    assert extract(detector_view(t_covered))["instruction_value_coverage"] == 1.0
    assert extract(detector_view(t_missing))["instruction_value_coverage"] == 0.0
    assert extract(detector_view(t_no_tokens))["instruction_value_coverage"] == 1.0
