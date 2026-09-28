"""Documentation stays internally consistent.

Two cheap guards: every relative Markdown link in the docs actually points
somewhere, and the shipped `claimcheck.toml.example` is valid configuration
input, not just a plausible-looking text file.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from agent_claimcheck.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Files (or globs, relative to the repo root) whose Markdown links are checked.
_CHECKED_GLOBS: tuple[str, ...] = (
    "docs/*.md",
    "docs/adr/*.md",
    "examples/README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
)

#: `[text](target)` / `[text](target "title")`, not preceded by `!` (an image
#: is still a real link to check, so this only matters for readability).
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _checked_files() -> list[Path]:
    files: list[Path] = []
    for pattern in _CHECKED_GLOBS:
        files.extend(sorted(REPO_ROOT.glob(pattern)))
    return files


def _relative_link_targets(text: str) -> list[str]:
    targets = []
    for target in _LINK_RE.findall(text):
        if target.startswith("#"):
            continue  # same-page anchor, not a file reference
        scheme = urlsplit(target).scheme
        if scheme in ("http", "https", "mailto"):
            continue
        targets.append(target)
    return targets


@pytest.mark.parametrize("md_file", _checked_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_relative_markdown_links_resolve(md_file: Path) -> None:
    text = md_file.read_text(encoding="utf-8")
    for target in _relative_link_targets(text):
        path_part = target.split("#", 1)[0]
        resolved = (md_file.parent / path_part).resolve()
        assert resolved.exists(), f"{md_file}: link target {target!r} does not exist ({resolved})"


def test_at_least_one_file_was_checked() -> None:
    # A guard against the globs above silently matching nothing (e.g. after
    # a directory rename), which would make the parametrized test a no-op.
    assert len(_checked_files()) >= 5


def test_claimcheck_toml_example_loads_with_the_config_loader() -> None:
    example = REPO_ROOT / "claimcheck.toml.example"
    cfg = load_config(example)

    assert cfg.gate.verified == 0.80
    assert cfg.gate.false_success == 0.20
    assert cfg.judge.base_url == "https://openrouter.ai/api/v1"
    assert cfg.judge.api_key_env == "CLAIMCHECK_API_KEY"
    assert cfg.judge.temperature == 0.0
    assert cfg.judge.max_tokens == 400
    assert cfg.judge.json_mode is True
    assert cfg.judge.timeout_s == 60.0
    assert cfg.judge.concurrency == 8
    assert cfg.budget.max_usd == 1.0
    assert cfg.rules.packs == ()
    assert cfg.rules.non_success_types == (
        "failed",
        "blocked",
        "gave_up",
        "needs_input",
        "partial",
    )
