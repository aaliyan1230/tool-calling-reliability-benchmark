from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcrb.v03.analysis import analyze_run
from tcrb.v03.cases import build_base_cases, build_case_variants, validate_case_variants
from tcrb.v03.prompts import prompt_bundle
from tcrb.v03.providers import ProviderResponse, parse_monitor_result
from tcrb.v03.runner import build_call_specs, prepare_dataset, run_stage


def test_fixed_dataset_is_balanced_blinded_and_matched() -> None:
    cases = build_base_cases()
    variants = build_case_variants(cases)

    assert len(cases) == 16
    assert len(variants) == 64
    assert validate_case_variants(variants) == []
    assert {
        domain: sum(case.domain == domain for case in cases)
        for domain in {case.domain for case in cases}
    } == {
        "customer_support": 4,
        "ecommerce": 4,
        "fintech": 4,
        "developer_tools": 4,
    }


def test_verifier_prompt_does_not_contain_private_gold() -> None:
    variant = build_case_variants()[0]
    bundle = prompt_bundle(variant.verifier_view, "baseline")
    serialized = json.dumps(bundle)

    assert "expected_flag" not in serialized
    assert "payload_state" not in serialized
    assert variant.private_gold.corruption_reason not in serialized


@pytest.mark.parametrize(
    ("text", "decision"),
    [
        ('{"decision":"allow","confidence":0.8,"evidence_ids":["x"],"reason":"ok"}', "allow"),
        ('```json\n{"decision":"block","confidence":1,"evidence_ids":["x"],"reason":"mismatch"}\n```', "block"),
        ('prefix {"decision":"review","confidence":0.5,"evidence_ids":[],"reason":"unclear"} suffix', "review"),
    ],
)
def test_monitor_result_parser_accepts_supported_json_shapes(text: str, decision: str) -> None:
    assert parse_monitor_result(text).decision == decision


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"decision":"yes","confidence":0.8,"evidence_ids":[],"reason":"x"}',
        '{"decision":"allow","confidence":2,"evidence_ids":[],"reason":"x"}',
        '{"decision":"allow","confidence":0.8,"evidence_ids":"x","reason":"x"}',
    ],
)
def test_monitor_result_parser_rejects_invalid_results(text: str) -> None:
    with pytest.raises(ValueError):
        parse_monitor_result(text)


def test_run_matrices_have_locked_sizes() -> None:
    providers = ("deepseek", "gpt")
    assert len(build_call_specs(stage="smoke", providers=providers)) == 32
    assert len(build_call_specs(stage="core", providers=providers)) == 256
    assert len(build_call_specs(stage="stability", providers=providers)) == 128
    assert len(build_call_specs(stage="stress", providers=providers)) == 64


def test_call_ids_are_bound_to_prompt_content(monkeypatch: pytest.MonkeyPatch) -> None:
    before = build_call_specs(stage="smoke", providers=("deepseek",))[0].call_id
    monkeypatch.setattr(
        "tcrb.v03.runner.prompt_bundle",
        lambda view, prompt_variant: {"system": "changed", "user": "changed"},
    )
    after = build_call_specs(stage="smoke", providers=("deepseek",))[0].call_id
    assert before != after


def test_prepare_keeps_verifier_views_and_private_gold_separate(tmp_path: Path) -> None:
    manifest = prepare_dataset(tmp_path, include_stress=True)

    public_text = (tmp_path / "verifier_views.jsonl").read_text(encoding="utf-8")
    private_text = (tmp_path / "private_gold.jsonl").read_text(encoding="utf-8")
    assert manifest["private_gold_separate"] is True
    assert len(public_text.splitlines()) == 64
    assert len(private_text.splitlines()) == 64
    assert "expected_flag" not in public_text
    assert "expected_flag" in private_text


def test_runner_resumes_successful_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return ProviderResponse(
            provider="fake",
            model_id="fake-model",
            output_text='{"decision":"allow","confidence":0.9,"evidence_ids":["order_record"],"reason":"supported"}',
            raw_response={"id": calls},
            usage={"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 0},
            latency_ms=1,
            estimated_extra_cost_usd=0,
        )

    monkeypatch.setattr("tcrb.v03.runner.call_monitor", fake_call)
    first = run_stage(tmp_path, stage="smoke", providers=("deepseek",), max_retries=0)
    second = run_stage(tmp_path, stage="smoke", providers=("deepseek",), max_retries=0)

    assert first["completed_now"] == 16
    assert second["completed_now"] == 0
    assert second["skipped_cached"] == 16
    assert calls == 16


def test_analysis_computes_expected_label_effect(tmp_path: Path) -> None:
    prepare_dataset(tmp_path)
    variants = build_case_variants()
    rows = []
    for index, variant in enumerate(variants):
        gold = variant.private_gold
        # Artificial monitor: always trusts the self-label. It flags warnings and
        # allows verified outputs, regardless of the actual payload.
        decision = "review" if gold.self_label == "warning" else "allow"
        rows.append(
            {
                "call_id": f"call-{index}",
                "case_id": gold.case_id,
                "base_case_id": gold.base_case_id,
                "domain": gold.domain,
                "provider": "deepseek",
                "prompt_variant": "baseline",
                "repeat_index": 0,
                "exploratory": False,
                "status": "success",
                "model_id": "fake",
                "monitor_result": {
                    "decision": decision,
                    "confidence": 0.9,
                    "evidence_ids": [gold.relevant_independent_evidence_ids[0]],
                    "reason": "synthetic test",
                },
                "latency_ms": 1,
                "usage": {},
                "estimated_extra_cost_usd": 0,
            }
        )
    (tmp_path / "responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = analyze_run(tmp_path, bootstrap_samples=100)
    metrics = summary["metrics"]["deepseek:baseline"]
    assert metrics["self_label_effect"] == 1.0
    assert metrics["corruption_sensitivity"] == 0.0
    assert metrics["corruption_detection_rate"] == 0.5
    assert metrics["false_alarm_rate"] == 0.5


def test_analysis_keeps_only_newest_prompt_attempt_per_cell(tmp_path: Path) -> None:
    prepare_dataset(tmp_path)
    variant = build_case_variants()[0]
    base = {
        "case_id": variant.private_gold.case_id,
        "base_case_id": variant.private_gold.base_case_id,
        "domain": variant.private_gold.domain,
        "provider": "deepseek",
        "prompt_variant": "baseline",
        "repeat_index": 0,
        "exploratory": False,
        "status": "success",
        "model_id": "fake",
        "latency_ms": 1,
        "usage": {},
        "estimated_extra_cost_usd": 0,
    }
    old_invalid = {**base, "call_id": "old", "monitor_result": None, "parse_error": "missing field"}
    new_valid = {
        **base,
        "call_id": "new",
        "monitor_result": {
            "decision": "allow",
            "confidence": 0.9,
            "evidence_ids": ["order_record"],
            "reason": "supported",
        },
    }
    (tmp_path / "responses.jsonl").write_text(
        json.dumps(old_invalid) + "\n" + json.dumps(new_valid) + "\n",
        encoding="utf-8",
    )

    summary = analyze_run(tmp_path, bootstrap_samples=10)
    assert summary["rows"] == 1
    assert summary["responses_by_status"] == {"valid": 1, "invalid_or_error": 0}
