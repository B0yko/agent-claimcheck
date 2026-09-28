"""Configuration precedence: CLI > env > TOML > default."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_claimcheck.config import (
    DEFAULT_BASE_URL,
    cache_dir,
    ledger_path,
    load_config,
    resolve_api_key,
)


@pytest.fixture
def toml_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "claimcheck.toml"
    path.write_text(
        """
[gate]
verified = 0.85

[budget]
max_usd = 3.0

[judge]
model = "toml-model"
base_url = "https://toml.example.com/v1"
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return path


def test_default_values_with_no_toml_no_env_no_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no claimcheck.toml here
    monkeypatch.delenv("CLAIMCHECK_MAX_USD", raising=False)
    monkeypatch.delenv("CLAIMCHECK_MODEL", raising=False)
    monkeypatch.delenv("CLAIMCHECK_BASE_URL", raising=False)

    cfg = load_config()
    assert cfg.budget.max_usd == 1.0
    assert cfg.judge.model is None
    assert cfg.judge.base_url == DEFAULT_BASE_URL
    assert cfg.gate.verified == 0.80


def test_toml_overrides_default(toml_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAIMCHECK_MAX_USD", raising=False)
    monkeypatch.delenv("CLAIMCHECK_MODEL", raising=False)

    cfg = load_config()
    assert cfg.budget.max_usd == 3.0
    assert cfg.judge.model == "toml-model"
    assert cfg.gate.verified == 0.85


def test_env_overrides_toml(toml_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMCHECK_MAX_USD", "5.0")
    monkeypatch.setenv("CLAIMCHECK_MODEL", "env-model")

    cfg = load_config()
    assert cfg.budget.max_usd == 5.0
    assert cfg.judge.model == "env-model"
    # Untouched by env: TOML value still wins over the default.
    assert cfg.gate.verified == 0.85


def test_cli_overrides_env_and_toml(toml_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIMCHECK_MAX_USD", "5.0")
    monkeypatch.setenv("CLAIMCHECK_MODEL", "env-model")

    cfg = load_config(cli={"budget.max_usd": 9.0, "judge.model": "cli-model"})
    assert cfg.budget.max_usd == 9.0
    assert cfg.judge.model == "cli-model"


def test_explicit_config_path_overrides_default_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    custom = other_dir / "custom.toml"
    custom.write_text("[budget]\nmax_usd = 7.0\n", encoding="utf-8")

    cfg = load_config(config_path=custom)
    assert cfg.budget.max_usd == 7.0


def test_cache_dir_uses_xdg_cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cache_dir() == tmp_path / "agent-claimcheck"


def test_cache_dir_falls_back_to_home_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert cache_dir() == Path.home() / ".cache" / "agent-claimcheck"


def test_ledger_path_expands_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAIMCHECK_LEDGER", raising=False)
    assert ledger_path("~/my-ledger.jsonl") == Path.home() / "my-ledger.jsonl"


def test_ledger_path_defaults_to_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAIMCHECK_LEDGER", raising=False)
    assert ledger_path() == cache_dir() / "ledger.jsonl"


def test_resolve_api_key_prefers_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert resolve_api_key("https://openrouter.ai/api/v1", explicit="direct-key") == "direct-key"


def test_resolve_api_key_falls_back_to_openrouter_on_openrouter_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLAIMCHECK_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert resolve_api_key("https://openrouter.ai/api/v1") == "or-key"


def test_resolve_api_key_never_falls_back_for_other_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAIMCHECK_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert resolve_api_key("https://api.other-provider.example/v1") is None
