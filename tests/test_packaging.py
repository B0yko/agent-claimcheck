"""The wheel must carry byte-identical copies of the canonical data files."""

from __future__ import annotations

import subprocess
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Iterator[zipfile.ZipFile]:
    out_dir = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        yield archive


def test_packaged_schema_is_byte_identical_to_repo_schema(wheel: zipfile.ZipFile) -> None:
    canonical = (REPO_ROOT / "schemas" / "agent-trace-v1.json").read_bytes()
    packaged = wheel.read("agent_claimcheck/schemas/agent-trace-v1.json")
    assert packaged == canonical


def test_rule_packs_are_included_in_the_wheel(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())
    packs_dir = REPO_ROOT / "src" / "agent_claimcheck" / "rules" / "packs"
    for pack in ("booking", "crm", "coding", "generic"):
        packaged_path = f"agent_claimcheck/rules/packs/{pack}.yaml"
        assert packaged_path in names
        assert wheel.read(packaged_path) == (packs_dir / f"{pack}.yaml").read_bytes()
