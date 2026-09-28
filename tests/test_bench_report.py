"""Tests for `bench/report.py`: purely-from-recorded-files report building."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from factory import numbers_in

from agent_claimcheck.bench.report import (
    README_END,
    README_START,
    _hypothesis_h4,
    build_report,
    check_readme_diff,
    histogram_svg_for,
    reliability_svg_for,
    verify_judge_requests,
)
from agent_claimcheck.calibration import CalibratorSet, fit_calibrator
from agent_claimcheck.claims import ClaimExtractor
from agent_claimcheck.config import DEFAULT_BASE_URL
from agent_claimcheck.judge.render import (
    JudgeSpec,
    load_prompt,
    openrouter_extra_body,
    render_request,
    request_sha256,
)
from agent_claimcheck.redact import detector_view, resolve_claims
from agent_claimcheck.rules.engine import builtin_packs
from agent_claimcheck.schema import load_traces

# --- offline-only report -----------------------------------------------------


@pytest.fixture(scope="session")
def offline_report(offline_recorded_dir: Path) -> tuple[dict[str, Any], str]:
    return build_report(offline_recorded_dir)


def test_dataset_line_counts(offline_report: tuple[dict[str, Any], str]) -> None:
    bench_json, _ = offline_report
    d = bench_json["dataset"]
    assert d == {
        "n_total": 300,
        "n_train": 180,
        "n_test": 120,
        "n_false_test": 48,
        "n_domains": 3,
        "n_false_kinds": 7,
        "seed": 20260924,
        "test_sha256_short": "b" * 12,
    }


def test_table_b_offline_detectors(offline_report: tuple[dict[str, Any], str]) -> None:
    bench_json, _ = offline_report
    detectors = bench_json["detectors"]
    assert set(detectors) == {
        "trust-agent",
        "any-error",
        "rules",
        "classifier-lr",
        "cascade-offline",
    }
    for key in ("rules", "classifier-lr", "cascade-offline"):
        d = detectors[key]["decisions"]
        # caught + missed covers decided failures; the rest went to review.
        undecided_failures = d["confusion"]["failure"]["unverifiable"]
        assert d["caught"] + d["missed"] + undecided_failures == d["n_failure"] == 48
        assert d["n_success"] == 72
        assert 0 <= d["caught"] <= 48
        assert 0 <= d["false_alarms"] <= 72

    # trust-agent always says "verified": it catches nothing.
    assert detectors["trust-agent"]["decisions"]["missed"] == 48
    assert detectors["trust-agent"]["decisions"]["caught"] == 0


def test_cascade_offline_never_worse_covered_than_rules_alone(
    offline_report: tuple[dict[str, Any], str],
) -> None:
    bench_json, _ = offline_report
    detectors = bench_json["detectors"]
    assert (
        detectors["cascade-offline"]["decisions"]["coverage"]
        >= detectors["rules"]["decisions"]["coverage"]
    )


def test_hypotheses_h1_and_h3_are_computed_offline(
    offline_report: tuple[dict[str, Any], str],
) -> None:
    bench_json, _ = offline_report
    h1, h3 = bench_json["hypotheses"]["h1"], bench_json["hypotheses"]["h3"]
    assert isinstance(h1["supported"], bool)
    assert "missed" in h1 and "coverage" in h1
    assert isinstance(h3["supported"], bool)
    assert set(h3["domains"]) == {"booking", "crm", "coding"}


def test_hypotheses_h2_and_h4_are_na_offline(offline_report: tuple[dict[str, Any], str]) -> None:
    bench_json, _ = offline_report
    h2, h4 = bench_json["hypotheses"]["h2"], bench_json["hypotheses"]["h4"]
    assert h2 == {"supported": False, "detail": "n/a (offline run)"}
    assert h4 == {"supported": False, "detail": "n/a (offline run)"}


def test_judge_sections_are_na_offline(offline_report: tuple[dict[str, Any], str]) -> None:
    bench_json, bench_md = offline_report
    assert bench_json["ablation"] is None
    assert bench_json["parse_error_rates"] is None
    assert "n/a (offline run)" in bench_md


def test_recall_by_kind_totals(offline_report: tuple[dict[str, Any], str]) -> None:
    bench_json, _ = offline_report
    rules_recall = bench_json["recall_by_kind"]["rules"]
    assert sum(cell["total"] for cell in rules_recall.values()) == 48


def test_leakage_number_is_the_frozen_constant(offline_report: tuple[dict[str, Any], str]) -> None:
    bench_json, bench_md = offline_report
    assert bench_json["leakage_test_auroc"] == 0.434
    assert "0.434" in bench_md


def test_findings_numbers_all_appear_in_bench_md(
    offline_report: tuple[dict[str, Any], str],
) -> None:
    _, bench_md = offline_report
    findings = bench_md.split("### Findings")[1]
    before_findings = bench_md.split("### Findings")[0]
    for number in numbers_in(findings):
        assert number in before_findings, f"{number!r} in Findings is not backed by a table"


def test_verify_judge_requests_empty_without_judge_records(offline_recorded_dir: Path) -> None:
    assert verify_judge_requests(offline_recorded_dir) == []


def test_check_readme_diff(offline_report: tuple[dict[str, Any], str], tmp_path: Path) -> None:
    _, bench_md = offline_report
    readme = tmp_path / "README.md"

    readme.write_text(f"# x\n\n{README_START}\nstale\n{README_END}\n", encoding="utf-8")
    diff = check_readme_diff(readme, bench_md)
    assert diff is not None
    assert "stale" in diff

    readme.write_text(
        f"# x\n\n{README_START}\n{bench_md.strip()}\n{README_END}\n", encoding="utf-8"
    )
    assert check_readme_diff(readme, bench_md) is None


def test_check_readme_diff_missing_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# no markers here\n", encoding="utf-8")
    diff = check_readme_diff(readme, "anything")
    assert diff is not None
    assert "missing" in diff


def test_svg_panels_are_deterministic_and_valid(offline_recorded_dir: Path) -> None:
    import xml.dom.minidom as minidom

    reliability = reliability_svg_for(offline_recorded_dir)
    histogram = histogram_svg_for(offline_recorded_dir)
    assert reliability == reliability_svg_for(offline_recorded_dir)
    minidom.parseString(reliability)
    minidom.parseString(histogram)
    assert 'fill="#ffffff"' in reliability  # explicit white background
    assert "rules" in reliability
    assert "classifier-lr" in reliability


# --- with a synthetic judge-records fixture ----------------------------------


def _pseudo_p(trace_id: str, model: str, prompt_name: str, outcome: str, injection: str) -> float:
    import hashlib

    h = int(hashlib.sha256(f"{trace_id}|{model}|{prompt_name}".encode()).hexdigest(), 16)
    frac = (h % 1000) / 1000.0
    if outcome == "success":
        return 0.97 + 0.02 * frac
    if injection == "reviewer_injection":
        return 0.6 + 0.35 * frac  # fooled: skews high despite being a failure
    return 0.005 + 0.02 * frac


@pytest.fixture(scope="session")
def judge_recorded_dir(
    offline_recorded_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """A copy of the offline recorded run with a synthetic `judge-records.jsonl`,
    real prompt/request sha256s, and matching calibrators + `run.json` prices,
    exercising every judge-dependent report section.
    """
    dest = tmp_path_factory.mktemp("judge-recorded")
    shutil.copytree(offline_recorded_dir, dest, dirs_exist_ok=True)

    extractor = ClaimExtractor(list(builtin_packs().values()))
    traces_by_split = {"train": load_traces("bench:train"), "test": load_traces("bench:test")}
    views = {}
    ground_truth = {}
    for split, traces in traces_by_split.items():
        for t in traces:
            views[(t.trace_id, split)] = resolve_claims(detector_view(t), extractor)
            assert t.ground_truth is not None
            ground_truth[t.trace_id] = {
                "outcome": t.ground_truth.outcome,
                "injection": t.ground_truth.details.get("injection", "none"),
            }

    models = ["fake/vendor-a", "fake/vendor-b", "fake/vendor-c"]
    # Deliberately not all the same, so a replay that assumes one fixed
    # json_mode/supports_reasoning for every judge cannot pass by accident.
    model_flags = {
        "fake/vendor-a": {"json_mode": True, "supports_reasoning": False},
        "fake/vendor-b": {"json_mode": False, "supports_reasoning": False},
        "fake/vendor-c": {"json_mode": True, "supports_reasoning": True},
    }
    prompts = {name: load_prompt(name) for name in ("claim-audit", "claim-by-claim")}

    records: list[dict[str, Any]] = []
    train_p_by_model: dict[str, tuple[list[float], list[int]]] = {m: ([], []) for m in models}

    for model in models:
        flags = model_flags[model]
        extra_body = openrouter_extra_body(DEFAULT_BASE_URL, **flags)
        prompt_names = ["claim-audit", "claim-by-claim"] if model == models[0] else ["claim-audit"]
        for prompt_name in prompt_names:
            prompt = prompts[prompt_name]
            spec = JudgeSpec(
                model=model,
                temperature=0.0,
                max_tokens=400,
                json_mode=flags["json_mode"],
                extra_body=extra_body,
            )
            detector_key = (
                f"judge:{model}" if prompt_name == "claim-audit" else f"judge:{model}:{prompt_name}"
            )
            for split, traces in traces_by_split.items():
                for idx, t in enumerate(traces):
                    view = views[(t.trace_id, split)]
                    body = render_request(view, prompt, spec)
                    gt = ground_truth[t.trace_id]
                    p_raw = _pseudo_p(
                        t.trace_id, model, prompt_name, gt["outcome"], gt["injection"]
                    )
                    # Half the test-split calls are cache hits: free and
                    # near-instant, so cost/latency stats must exclude them.
                    cached = split == "test" and idx % 2 == 0
                    records.append(
                        {
                            "detector": detector_key,
                            "model": model,
                            "prompt_name": prompt_name,
                            "prompt_version": prompt.version,
                            "prompt_sha256": prompt.sha256,
                            "request_sha256": request_sha256(body),
                            "trace_id": t.trace_id,
                            "split": split,
                            "parsed": None,
                            "p_raw": p_raw,
                            "abstain": False,
                            "abstain_reason": None,
                            "raw_text": "{}",
                            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                            "cost_usd": 0.0 if cached else 0.0005,
                            "latency_ms": 5.0 if cached else 800.0,
                            "cached": cached,
                            "attempts": 1,
                            "invalid_citation": False,
                        }
                    )
                    if split == "train" and prompt_name == "claim-audit":
                        train_p_by_model[model][0].append(p_raw)
                        train_p_by_model[model][1].append(1 if gt["outcome"] == "success" else 0)

    with (dest / "judge-records.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")

    calibrators = CalibratorSet.load(dest / "calibration.json")
    new_calibrators = dict(calibrators.calibrators)
    for model in models:
        p, labels = train_p_by_model[model]
        new_calibrators[f"judge:{model}"] = fit_calibrator(
            f"judge:{model}", p, labels, fitted_on="bench:train"
        )
    CalibratorSet(
        version=calibrators.version,
        fitted_on=calibrators.fitted_on,
        base_rate=calibrators.base_rate,
        calibrators=new_calibrators,
    ).save(dest / "calibration.json")

    run_meta = json.loads((dest / "run.json").read_text(encoding="utf-8"))
    run_meta["judges"] = [
        {
            "id": "fake/vendor-a",
            "price_in_per_m": 0.10,
            "price_out_per_m": 0.30,
            **model_flags["fake/vendor-a"],
        },
        {
            "id": "fake/vendor-b",
            "price_in_per_m": 0.50,
            "price_out_per_m": 1.50,
            **model_flags["fake/vendor-b"],
        },
        {
            "id": "fake/vendor-c",
            "price_in_per_m": 1.00,
            "price_out_per_m": 3.00,
            **model_flags["fake/vendor-c"],
        },
    ]
    # Each judge ran at concurrency > 1, so its 300 calls overlapped: the
    # measured wall-clock time is far below the sum of the calls' own
    # latencies, and that recorded number (not the sum) is what the report
    # must scale into wall-clock seconds per 1,000 traces.
    run_meta["judge_wall_clock_s"] = {
        "judge:fake/vendor-a": 40.5,
        "judge:fake/vendor-b": 81.0,
        "judge:fake/vendor-c": 121.5,
    }
    (dest / "run.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n", "utf-8")
    return dest


def test_verify_judge_requests_passes_on_untampered_records(judge_recorded_dir: Path) -> None:
    assert verify_judge_requests(judge_recorded_dir) == []


def test_tampered_request_sha256_is_caught(judge_recorded_dir: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(judge_recorded_dir, tampered)
    lines = (tampered / "judge-records.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["request_sha256"] = "0" * 64
    lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    (tampered / "judge-records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    mismatches = verify_judge_requests(tampered)
    assert len(mismatches) == 1
    assert "request_sha256 mismatch" in mismatches[0]


def test_tampered_prompt_sha256_is_caught(judge_recorded_dir: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "tampered-prompt"
    shutil.copytree(judge_recorded_dir, tampered)
    lines = (tampered / "judge-records.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["prompt_sha256"] = "0" * 64
    lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    (tampered / "judge-records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    mismatches = verify_judge_requests(tampered)
    assert len(mismatches) == 1
    assert "prompt_sha256 mismatch" in mismatches[0]


@pytest.fixture(scope="session")
def judge_report(judge_recorded_dir: Path) -> tuple[dict[str, Any], str]:
    return build_report(judge_recorded_dir)


def test_judge_detectors_and_cascade_appear_in_table_a_b(
    judge_report: tuple[dict[str, Any], str],
) -> None:
    bench_json, _ = judge_report
    detectors = bench_json["detectors"]
    for key in ("judge:fake/vendor-a", "judge:fake/vendor-b", "judge:fake/vendor-c", "cascade"):
        assert key in detectors
        assert detectors[key]["auroc"] is not None
        assert detectors[key]["decisions"]["n_failure"] == 48


def test_cascade_covers_every_test_trace(judge_report: tuple[dict[str, Any], str]) -> None:
    bench_json, _ = judge_report
    cascade_decisions = bench_json["detectors"]["cascade"]["decisions"]
    assert cascade_decisions["n"] == 120


def test_judge_cost_and_latency_stats_exclude_cache_hits(
    judge_report: tuple[dict[str, Any], str],
) -> None:
    bench_json, _ = judge_report
    stats = bench_json["detectors"]["judge:fake/vendor-a"]
    # 60 of the 120 test-split calls are cache hits (free, ~instant); the
    # other 60 cost $0.0005 and took 800 ms. Averaging over all 120 would
    # understate both figures.
    assert stats["usd_per_1k"] == pytest.approx(0.5)
    assert stats["p50_ms"] == pytest.approx(800.0)
    assert stats["p95_ms"] == pytest.approx(800.0)


def test_judge_wall_clock_comes_from_the_recorded_run_not_summed_latency(
    judge_report: tuple[dict[str, Any], str],
) -> None:
    bench_json, _ = judge_report
    stats = bench_json["detectors"]["judge:fake/vendor-a"]
    # run.json records 40.5 s of actual wall-clock time for this judge's 300
    # calls (it ran at concurrency > 1); summing the 60 fresh test-split
    # calls' own 800 ms latencies instead would give 800 s/1k, ~6x too high.
    assert stats["wall_s_per_1k"] == pytest.approx(135.0)


def test_h4_uses_mean_recall_over_the_six_other_kinds_not_pooled() -> None:
    """`reviewer_injection` recall must be compared against the mean of the
    per-kind recalls for the other six kinds, not a recall pooled across
    their rows. Here reviewer_injection catches 3/10 (0.3); one other kind
    catches 1/1 (1.0) and another 0/19 (0.0), so the per-kind mean is 0.5
    (0.3 < 0.5: fooled). Pooling those 20 rows together instead would give
    1/20 = 0.05, and 0.3 < 0.05 is false, hiding the effect.
    """

    def _row(injection: str, *, caught: bool) -> dict[str, Any]:
        return {
            "injection": injection,
            "outcome": "failure",
            "abstain": False,
            "p_raw": 0.1 if caught else 0.9,
        }

    rows = (
        [_row("reviewer_injection", caught=True) for _ in range(3)]
        + [_row("reviewer_injection", caught=False) for _ in range(7)]
        + [_row("phantom_action", caught=True)]
        + [_row("error_ignored", caught=False) for _ in range(19)]
    )
    result = _hypothesis_h4({"judge:fake-model": rows})
    detail = result["judges"]["judge:fake-model"]
    assert detail["reviewer_injection_recall"] == pytest.approx(0.3)
    assert detail["other_mean_recall"] == pytest.approx(0.5)
    assert detail["fooled"] is True
    assert result["supported"] is True


def test_h4_is_supported_when_reviewer_injection_fools_a_judge(
    judge_report: tuple[dict[str, Any], str],
) -> None:
    bench_json, _ = judge_report
    h4 = bench_json["hypotheses"]["h4"]
    assert h4["supported"] is True
    assert any(row["fooled"] for row in h4["judges"].values())


def test_h2_extremes_and_calibration_improvement(judge_report: tuple[dict[str, Any], str]) -> None:
    bench_json, _ = judge_report
    h2 = bench_json["hypotheses"]["h2"]
    assert set(h2["judges"]) == {
        "judge:fake/vendor-a",
        "judge:fake/vendor-b",
        "judge:fake/vendor-c",
    }
    for row in h2["judges"].values():
        assert row["extremes_raw"] > 0.5


def test_ablation_picks_the_cheapest_judge_model(judge_report: tuple[dict[str, Any], str]) -> None:
    bench_json, _ = judge_report
    ablation = bench_json["ablation"]
    assert ablation is not None
    assert ablation["model"] == "fake/vendor-a"
    assert "claim-audit" in ablation and "claim-by-claim" in ablation


def test_parse_error_rates_present_per_judge(judge_report: tuple[dict[str, Any], str]) -> None:
    bench_json, _ = judge_report
    rates = bench_json["parse_error_rates"]
    assert rates is not None
    assert set(rates) == {"judge:fake/vendor-a", "judge:fake/vendor-b", "judge:fake/vendor-c"}


def test_recall_by_kind_includes_judges_and_cascade(
    judge_report: tuple[dict[str, Any], str],
) -> None:
    bench_json, _ = judge_report
    recall = bench_json["recall_by_kind"]
    assert "judge:fake/vendor-a" in recall
    assert "cascade" in recall
    # every judge is fully fooled by the synthetic reviewer_injection bias.
    assert recall["judge:fake/vendor-a"]["reviewer_injection"]["caught"] == 0


def test_judge_findings_numbers_appear_in_bench_md(
    judge_report: tuple[dict[str, Any], str],
) -> None:
    bench_json, bench_md = judge_report
    assert bench_json["ablation"] is not None
    findings = bench_md.split("### Findings")[1]
    before_findings = bench_md.split("### Findings")[0]
    for number in numbers_in(findings):
        assert number in before_findings


def test_leakage_constant_matches_the_dataset_card() -> None:
    from agent_claimcheck.bench.report import LEAKAGE_TEST_AUROC

    card = (Path(__file__).resolve().parents[1] / "benchmark" / "v1" / "DATASET_CARD.md").read_text(
        encoding="utf-8"
    )
    assert f"{LEAKAGE_TEST_AUROC:.3f} AUROC** on test" in card
