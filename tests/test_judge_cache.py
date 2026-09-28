"""The on-disk judge cache: atomic writes, `judge/` layout."""

from __future__ import annotations

from pathlib import Path

from agent_claimcheck.judge.cache import JudgeCache, cache_key


def test_cache_key_is_deterministic_and_input_sensitive() -> None:
    a = cache_key("model-a", 1, '{"x":1}')
    b = cache_key("model-a", 1, '{"x":1}')
    c = cache_key("model-a", 2, '{"x":1}')
    d = cache_key("model-b", 1, '{"x":1}')
    assert a == b
    assert len({a, c, d}) == 3


def test_miss_returns_none(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path)
    assert cache.get("nope") is None


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path)
    key = cache_key("m", 1, "{}")
    cache.put(key, "the response", {"prompt_tokens": 3, "completion_tokens": 4})
    hit = cache.get(key)
    assert hit is not None
    assert hit.content == "the response"
    assert hit.usage == {"prompt_tokens": 3, "completion_tokens": 4}


def test_cache_lives_under_judge_subdirectory(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path)
    key = cache_key("m", 1, "{}")
    cache.put(key, "x", {})
    assert (tmp_path / "judge" / f"{key}.json").exists()


def test_put_writes_atomically_no_leftover_tempfiles(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path)
    key = cache_key("m", 1, "{}")
    cache.put(key, "x", {})
    entries = list((tmp_path / "judge").iterdir())
    assert entries == [tmp_path / "judge" / f"{key}.json"]


def test_put_overwrites_existing_entry(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path)
    key = cache_key("m", 1, "{}")
    cache.put(key, "first", {})
    cache.put(key, "second", {})
    hit = cache.get(key)
    assert hit is not None
    assert hit.content == "second"


def test_corrupt_entry_treated_as_miss(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path)
    key = cache_key("m", 1, "{}")
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir(parents=True)
    (judge_dir / f"{key}.json").write_text("not json", encoding="utf-8")
    assert cache.get(key) is None
