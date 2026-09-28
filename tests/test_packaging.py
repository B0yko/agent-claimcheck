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


def test_result_schema_is_byte_identical_to_repo_schema(wheel: zipfile.ZipFile) -> None:
    canonical = (REPO_ROOT / "schemas" / "claimcheck-result-v1.json").read_bytes()
    packaged = wheel.read("agent_claimcheck/schemas/claimcheck-result-v1.json")
    assert packaged == canonical


def test_builtin_calibrators_are_included_in_the_wheel(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())
    model_path = REPO_ROOT / "src" / "agent_claimcheck" / "models" / "calibrators-v1.json"
    packaged_path = "agent_claimcheck/models/calibrators-v1.json"
    assert packaged_path in names
    assert wheel.read(packaged_path) == model_path.read_bytes()


def test_examples_traces_and_probes_are_included_in_the_wheel(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())
    examples_dir = REPO_ROOT / "examples"
    for filename in ("traces.jsonl", "probes.jsonl"):
        packaged_path = f"agent_claimcheck/_data/examples/{filename}"
        assert packaged_path in names
        assert wheel.read(packaged_path) == (examples_dir / filename).read_bytes()


def test_rule_packs_are_included_in_the_wheel(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())
    packs_dir = REPO_ROOT / "src" / "agent_claimcheck" / "rules" / "packs"
    for pack in ("booking", "crm", "coding", "generic"):
        packaged_path = f"agent_claimcheck/rules/packs/{pack}.yaml"
        assert packaged_path in names
        assert wheel.read(packaged_path) == (packs_dir / f"{pack}.yaml").read_bytes()


def test_lr_model_artifact_is_included_in_the_wheel(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())
    model_path = REPO_ROOT / "src" / "agent_claimcheck" / "models" / "lr-v1.json"
    packaged_path = "agent_claimcheck/models/lr-v1.json"
    assert packaged_path in names
    assert wheel.read(packaged_path) == model_path.read_bytes()


def test_benchmark_v1_is_included_in_the_wheel(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())
    bench_dir = REPO_ROOT / "benchmark" / "v1"
    for filename in ("traces.train.jsonl", "traces.test.jsonl", "manifest.json", "DATASET_CARD.md"):
        packaged_path = f"agent_claimcheck/_data/benchmark/v1/{filename}"
        assert packaged_path in names
        assert wheel.read(packaged_path) == (bench_dir / filename).read_bytes()


def test_judge_prompts_are_included_in_the_wheel(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())
    prompts_dir = REPO_ROOT / "src" / "agent_claimcheck" / "judge" / "prompts"
    for prompt in ("claim-audit", "claim-by-claim"):
        packaged_path = f"agent_claimcheck/judge/prompts/{prompt}.md"
        assert packaged_path in names
        assert wheel.read(packaged_path) == (prompts_dir / f"{prompt}.md").read_bytes()
