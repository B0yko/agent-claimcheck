"""Locate packaged data files, with aliases for the common ones.

Canonical files live at the repository root (``schemas/``, ``examples/``,
``benchmark/v1/`` and ``results/v0.1.0/``). They
are force-included into the wheel under ``agent_claimcheck/`` so the same
relative paths resolve for an installed package and for a checkout. Lookup
tries the packaged location first, then falls back to the repository root,
which is what makes editable installs work during development.
"""

from __future__ import annotations

from pathlib import Path

#: Alias name -> path relative to the repository root / package data root.
ALIASES: dict[str, str] = {
    "bench:train": "benchmark/v1/traces.train.jsonl",
    "bench:test": "benchmark/v1/traces.test.jsonl",
    "example:mixed": "examples/traces.jsonl",
    "example:browser": "examples/browser-demo.jsonl",
    "recorded:v0.1.0": "results/v0.1.0",
}

#: Top-level directory (first path segment) -> its location inside the
#: installed package, relative to the ``agent_claimcheck`` package directory.
_PACKAGE_SUBDIR: dict[str, str] = {
    "schemas": "schemas",
    "examples": "_data/examples",
    "benchmark": "_data/benchmark",
    "results": "_data/results",
}

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ResourceNotFoundError(FileNotFoundError):
    """Raised when neither the packaged copy nor the repo-root copy exists."""


def _packaged_candidate(rel: str) -> Path | None:
    top, _, rest = rel.partition("/")
    subdir = _PACKAGE_SUBDIR.get(top)
    if subdir is None:
        return None
    return _PACKAGE_DIR / Path(subdir) / rest if rest else _PACKAGE_DIR / Path(subdir)


def path(rel_or_alias: str) -> Path:
    """Resolve a packaged relative path or alias (e.g. ``example:browser``).

     Tries the packaged location under ``agent_claimcheck`` first, then the
     repository root. Raises ``ResourceNotFoundError`` with a clear message
     when a name is recognised but the underlying data does not exist yet
    .
    """
    rel = ALIASES.get(rel_or_alias, rel_or_alias)

    packaged = _packaged_candidate(rel)
    if packaged is not None and packaged.exists():
        return packaged

    repo_root_candidate = _REPO_ROOT / rel
    if repo_root_candidate.exists():
        return repo_root_candidate

    raise ResourceNotFoundError(
        f"resource not found: {rel_or_alias!r} "
        f"(looked for packaged data and under the repository root)"
    )
