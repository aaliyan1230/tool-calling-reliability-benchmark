from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcrb.v031.analysis import analyze_run
from tcrb.v031.audit import audit_run
from tcrb.v031.cases import build_tcrb_traces, selected_base_cases, validate_traces
from tcrb.v031.providers import parse_monitor_result
from tcrb.v031.runner import build_call_specs, prepare_dataset, run_stage
from tcrb.v031.schema import FOLLOWUP_VIEW_TYPES
from tcrb.v031.views import build_all_views, validate_views


def test_selected_cases_are_balanced_and_replay_valid() -> None:
    cases = selected_base_cases()
    traces = build_tcrb_traces()
    assert len(cases) == 12
    assert len(traces) == 24
    assert not validate_traces(traces)
    assert {trace.gold_outcome for trace in traces} == {"safe", "unsafe"}
    assert all(
        {trace.gold_outcome for trace in traces if trace.task_family == case.case_id}
        == {"safe", "unsafe"}
        for case in cases
    )


def test_views_change_only_visibility_and_protected_view_reports_missing_evidence() -> None:
    traces = build_tcrb_traces()
    pairs = build_all_views(traces)
    assert len(pairs) == 120
    assert not validate_views(traces, pairs)
    by_trace = {}
    for view, gold in pairs:
        by_trace.setdefault(view.trace_id, {})[gold.view_type] = view
    for trace in traces:
        cells = by_trace[trace.trace_id]
        assert [event["event_id"] for event in cells["full"].events] == [
            event.event_id for event in trace.events
        ]
        visible_silent = {event["event_id"] for event in cells["silent_filter"].events}
        assert len(set(trace.required_evidence_ids) - visible_silent) == 1
        assert cells["protected_summary"].completeness_manifest["missing_event_types"]


def test_public_view_does_not_reveal_outcome_or_required_evidence() -> None:
    view, gold = build_all_views(build_tcrb_traces()[:1])[0]
    public = json.dumps(view.to_public_dict())
    assert "gold_outcome" not in public
    assert "required_evidence_ids" not in public
    assert gold.gold_outcome == "safe"


def test_actionable_summary_projects_critical_values_without_private_labels() -> None:
    traces = build_tcrb_traces()
    pairs = build_all_views(traces, FOLLOWUP_VIEW_TYPES)
    assert len(pairs) == 144
    assert not validate_views(traces, pairs)
    actionable = [view for view, gold in pairs if gold.view_type == "actionable_summary"]
    assert len(actionable) == 24
    for view in actionable:
        summary = next(event for event in view.events if event["event_id"] == "actionable_summary")
        assert summary["payload"]["compression_policy"] == "structured_field_projection"
        assert "gold_outcome" not in json.dumps(view.to_public_dict())


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ('{"verdict":"approve","evidence_ids":["e1"],"reason":"supported"}', "approve"),
        ('```json\n{"verdict":"reject","evidence_ids":[],"reason":"conflict"}\n```', "reject"),
        ('{"verdict":"insufficient_evidence","evidence_ids":[],"reason":"missing"}', "insufficient_evidence"),
    ],
)
def test_monitor_parser(text: str, verdict: str) -> None:
    assert parse_monitor_result(text).verdict == verdict


def test_monitor_parser_rejects_invalid_payloads() -> None:
    for text in (
        "not json",
        '{"verdict":"allow","evidence_ids":[],"reason":"x"}',
        '{"verdict":"approve","evidence_ids":"e1","reason":"x"}',
        '{"verdict":"approve","evidence_ids":[],"reason":""}',
    ):
        with pytest.raises(ValueError):
            parse_monitor_result(text)


def test_locked_matrix_sizes(tmp_path: Path) -> None:
    prepare_dataset(tmp_path)
    providers = ("deepseek", "gpt")
    assert len(build_call_specs(tmp_path, stage="smoke", providers=providers)) == 40
    assert len(build_call_specs(tmp_path, stage="core", providers=providers)) == 240
    assert len(build_call_specs(tmp_path, stage="stability", providers=providers)) == 40


def test_actionable_matrix_size(tmp_path: Path) -> None:
    prepare_dataset(tmp_path, view_types=FOLLOWUP_VIEW_TYPES)
    assert len(build_call_specs(tmp_path, stage="core", providers=("deepseek", "gpt"))) == 288


def test_fake_results_score_paired_visibility_effect(tmp_path: Path) -> None:
    prepare_dataset(tmp_path)
    views = {
        row["view_id"]: row
        for row in (json.loads(line) for line in (tmp_path / "verifier_views.jsonl").read_text().splitlines())
    }
    gold = {
        row["view_id"]: row
        for row in (json.loads(line) for line in (tmp_path / "private_gold.jsonl").read_text().splitlines())
    }
    rows = []
    for index, (view_id, item) in enumerate(gold.items()):
        if item["view_type"] == "full":
            verdict = "reject" if item["gold_outcome"] == "unsafe" else "approve"
        elif item["view_type"] == "silent_filter":
            verdict = "approve" if item["gold_outcome"] == "unsafe" else "approve"
        else:
            verdict = "reject" if item["gold_outcome"] == "unsafe" else "approve"
        rows.append(
            {
                "call_id": f"fake-{index}",
                "provider": "deepseek",
                "model_id": "deepseek-v4-flash",
                "model_input": {"view_id": view_id},
                "repeat_index": 0,
                "status": "success",
                "monitor_result": {
                    "verdict": verdict,
                    "evidence_ids": [],
                    "reason": "fake",
                },
                "estimated_extra_cost_usd": 0,
            }
        )
    (tmp_path / "responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = analyze_run(tmp_path, bootstrap_samples=50)
    effect = summary["metrics"]["deepseek"]["paired_effects"]["silent_filter"]
    assert effect["unsafe_approval_delta"] == 1.0


def test_resume_skips_successful_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepare_dataset(tmp_path)
    calls = 0

    class FakeResponse:
        model_id = "fake-model"
        output_text = '{"verdict":"approve","evidence_ids":[],"reason":"ok"}'
        raw_response = {"fake": True}
        usage = {"input_tokens": 1, "output_tokens": 1, "cached_input_tokens": 0}
        latency_ms = 1
        estimated_extra_cost_usd = 0.0

    def fake_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setattr("tcrb.v031.runner.call_monitor", fake_call)
    first = run_stage(tmp_path, stage="smoke", providers=("deepseek",), max_retries=0)
    second = run_stage(tmp_path, stage="smoke", providers=("deepseek",), max_retries=0)
    assert first["completed_now"] == 20
    assert second["completed_now"] == 0
    assert second["skipped_cached"] == 20
    assert calls == 20


def test_audit_fails_before_calls_when_manifest_missing(tmp_path: Path) -> None:
    prepare_dataset(tmp_path)
    audit = audit_run(tmp_path, providers=("deepseek",))
    assert not audit["passed"]
    assert not audit["checks"]["locked_call_ids_present"]
