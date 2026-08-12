from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest
import tcrb.v034.augmentation as augmentation_module

from tcrb.v034.augmentation import (
    apply_patches,
    apply_reconciliation_patches,
    assistant_claim_units_after_target,
    build_packet,
    call_luna,
    differential_replay_delta,
    downstream_dependency_audit,
    editor_json_schema,
    enforce_call_spend_cap,
    luna_call_cost_upper_bound,
    planner_json_schema,
    public_trajectory,
    reconciler_json_schema,
    replay_source_fidelity,
    request_key,
    run_pilot,
    select_pilot_seeds,
    semantic_verifier_stages,
    semantic_verifier_json_schema,
    validate_reconciled_trajectory,
    validate_replayed_trajectory,
    validate_semantic_verdict,
    validate_tool_argument_constraints,
    validate_augmented_trajectory,
    validate_plan,
)


def fixture_trajectory() -> dict:
    return {
        "trajectory_id": "traj_fixture",
        "domain": "airline",
        "task_id": "999",
        "source_agent": "hidden",
        "events": [
            {"event_id": "u1", "role": "user", "content": "Book the flight."},
            {"event_id": "a1", "role": "assistant", "content": "Please confirm the $5 card charge."},
            {"event_id": "u2", "role": "user", "content": "Yes."},
            {"event_id": "c1", "role": "assistant", "tool_call": {"name": "book_reservation", "arguments": {"total": 255}}},
            {"event_id": "r1", "role": "tool", "call_event_id": "c1", "tool_result": {"content": "ok", "error": False}},
        ],
    }


def fixture_packet(trajectory: dict) -> dict:
    return {
        "trajectory": public_trajectory(trajectory),
        "policy": "Before a write, list details and obtain confirmation.",
        "policy_rules": [{"id": "airline.confirm_before_write", "text": "Before a write, list details and obtain confirmation."}],
        "state_changing_tool_contracts": {"book_reservation": "creates a reservation"},
        "write_event_ids_for_validation_only": ["c1"],
    }


def valid_plan() -> dict:
    return {
        "decision": "mutate",
        "selected_family": "changed_terms_after_confirmation",
        "target_rule_id": "airline.confirm_before_write",
        "target_write_event_id": "c1",
        "supporting_event_ids": ["a1", "u2", "c1"],
        "mutation_summary": "Remove the exact amount from the confirmation message while keeping the write.",
        "realism_reason": "Prices and confirmation wording can diverge in a long workflow.",
        "subtlety_reason": "The reviewer must compare the confirmation with the write.",
        "protected_facts": ["route", "user", "tool result"],
        "proposed_changes": [{"event_id": "a1", "operation": "replace_content", "field_or_scope": "confirmation amount", "intent": "omit the exact amount"}],
        "requires_environment_replay": False,
    }


def test_pilot_seeds_are_safe_and_outside_shortlist() -> None:
    seeds = select_pilot_seeds()
    assert [(row["domain"], row["trajectory"]["task_id"]) for row in seeds] == [
        ("airline", "17"),
        ("airline", "18"),
        ("retail", "104"),
        ("retail", "112"),
    ]
    assert all(row["write_event_ids"] for row in seeds)


def test_packet_excludes_source_and_gold_metadata(tmp_path: Path) -> None:
    trajectory = fixture_trajectory()
    packet = fixture_packet(trajectory)
    encoded = json.dumps(packet)
    assert "source_agent" not in encoded
    assert "task_id" not in encoded
    assert "reward" not in encoded
    assert "db_match" not in encoded
    assert "label" not in encoded


def test_plan_validator_accepts_real_target_and_rejects_unknown_rule() -> None:
    trajectory = fixture_trajectory()
    packet = fixture_packet(trajectory)
    assert validate_plan(valid_plan(), trajectory, packet) == []
    bad = valid_plan()
    bad["target_rule_id"] = "not-a-rule"
    assert "target_rule_id is not a supplied policy rule" in validate_plan(bad, trajectory, packet)


def test_content_patch_preserves_tool_result_and_validates() -> None:
    trajectory = fixture_trajectory()
    plan = valid_plan()
    editor = {
        "decision": "apply",
        "patches": [{"operation": "replace_content", "event_id": "a1", "new_content": "Please confirm the booking.", "new_arguments_json": None, "event_ids": [], "new_order": [], "reason": "omit amount"}],
        "requires_environment_replay": False,
        "changed_event_ids": ["a1"],
        "violation_explanation": "The write is no longer preceded by a complete amount listing.",
    }
    augmented, changed = apply_patches(trajectory, editor)
    assert changed == ["a1"]
    assert augmented["events"][-1]["tool_result"] == trajectory["events"][-1]["tool_result"]
    audit = validate_augmented_trajectory(trajectory, augmented, plan, editor, changed)
    assert audit["passed"], audit


def test_tool_result_edits_are_rejected() -> None:
    trajectory = fixture_trajectory()
    editor = {
        "decision": "apply",
        "patches": [{"operation": "replace_content", "event_id": "r1", "new_content": "tampered", "new_arguments_json": None, "event_ids": [], "new_order": [], "reason": "bad"}],
        "requires_environment_replay": False,
        "changed_event_ids": ["r1"],
    }
    with pytest.raises(ValueError):
        apply_patches(trajectory, editor)


def test_argument_patch_requires_replay_marker_in_audit() -> None:
    trajectory = fixture_trajectory()
    plan = valid_plan()
    plan["proposed_changes"] = [{"event_id": "c1", "operation": "replace_tool_arguments", "field_or_scope": "total", "intent": "change amount"}]
    editor = {
        "decision": "apply",
        "patches": [{"operation": "replace_tool_arguments", "event_id": "c1", "new_content": None, "new_arguments_json": "{\"total\": 384}", "event_ids": [], "new_order": [], "reason": "price changed"}],
        "requires_environment_replay": True,
        "changed_event_ids": ["c1"],
    }
    augmented, changed = apply_patches(trajectory, editor)
    audit = validate_augmented_trajectory(trajectory, augmented, plan, editor, changed)
    assert audit["passed"], audit
    assert audit["requires_environment_replay"]


def test_duplicate_event_ids_are_rejected() -> None:
    trajectory = fixture_trajectory()
    trajectory["events"].append({"event_id": "u1", "role": "user", "content": "duplicate"})
    with pytest.raises(ValueError, match="duplicate event ID"):
        apply_patches(trajectory, {"patches": []})


def test_structured_schemas_are_strict() -> None:
    assert planner_json_schema()["additionalProperties"] is False
    assert editor_json_schema()["additionalProperties"] is False
    assert reconciler_json_schema()["additionalProperties"] is False
    assert semantic_verifier_json_schema()["additionalProperties"] is False


def test_reconciler_only_repairs_later_assistant_prose() -> None:
    trajectory = fixture_trajectory()
    trajectory["events"].append({"event_id": "a2", "role": "assistant", "content": "The booking was created for $255."})
    reconciler = {
        "decision": "apply",
        "patches": [{"operation": "replace_content", "event_id": "a2", "new_content": "The booking was created for $384.", "reason": "match the replayed result"}],
        "changed_event_ids": ["a2"],
        "reason": "The later summary repeated the changed result.",
    }
    repaired, changed = apply_reconciliation_patches(trajectory, reconciler, ["c1", "r1"])
    assert changed == ["a2"]
    audit = validate_reconciled_trajectory(trajectory, repaired, reconciler, ["c1", "r1"], changed)
    assert audit["passed"], audit


def test_reconciler_rejects_user_or_tool_edits() -> None:
    trajectory = fixture_trajectory()
    reconciler = {
        "decision": "apply",
        "patches": [{"operation": "replace_content", "event_id": "r1", "new_content": "tampered", "reason": "bad"}],
        "changed_event_ids": ["r1"],
    }
    with pytest.raises(ValueError):
        apply_reconciliation_patches(trajectory, reconciler, ["c1", "r1"])


def test_differential_replay_ignores_preexisting_drift_and_reports_causal_delta() -> None:
    baseline = [
        {"event_id": "c1", "role": "assistant", "tool_call": {"name": "write", "arguments": {"x": 1}}},
        {"event_id": "r1", "role": "tool", "call_event_id": "c1", "tool_result": {"content": '{"value": 1}', "error": False}},
        {"event_id": "c2", "role": "assistant", "tool_call": {"name": "read", "arguments": {}}},
        {"event_id": "r2", "role": "tool", "call_event_id": "c2", "tool_result": {"content": '{"value": 7}', "error": False}},
    ]
    mutated = copy.deepcopy(baseline)
    mutated[1]["tool_result"]["content"] = '{"value": 2}'
    mutated[3]["tool_result"]["content"] = '{"value": 8}'
    delta = differential_replay_delta(baseline, mutated, "c1")
    assert delta["passed"]
    assert delta["pre_target_differences"] == []
    assert delta["causal_changed_event_ids"] == ["c1", "r1", "c2", "r2"]
    assert delta["causal_deltas"][0]["event_id"] == "r1"


def test_baseline_source_fidelity_uses_semantic_json_comparison() -> None:
    source = [{"event_id": "r1", "role": "tool", "tool_result": {"content": '{"a": 1, "b": 2}', "error": False}}]
    baseline = [{"event_id": "r1", "role": "tool", "tool_result": {"content": '{"b":2,"a":1}', "error": False}}]
    assert replay_source_fidelity(source, baseline)["passed"]
    baseline[0]["tool_result"]["content"] = '{"a": 1, "b": 3}'
    audit = replay_source_fidelity(source, baseline)
    assert not audit["passed"]
    assert audit["mismatched_result_event_ids"] == ["r1"]


def test_downstream_dependency_audit_rejects_stale_calculation() -> None:
    events = [
        {"event_id": "c1", "role": "assistant", "tool_call": {"name": "write", "arguments": {}}},
        {"event_id": "r1", "role": "tool", "call_event_id": "c1", "tool_result": {"content": "{}", "error": False}},
        {"event_id": "calc", "role": "assistant", "tool_call": {"name": "calculate", "arguments": {"expression": "6594 + 100"}}},
    ]
    deltas = [{
        "event_id": "r1",
        "baseline_result": {"content": {"refund": -6594}, "error": False},
        "mutated_result": {"content": {"refund": -6742}, "error": False},
    }]
    audit = downstream_dependency_audit(events, "c1", deltas)
    assert not audit["passed"]
    assert audit["issues"][0]["tool_event_id"] == "calc"
    events[-1]["tool_call"]["arguments"]["expression"] = "6742 + 100"
    assert downstream_dependency_audit(events, "c1", deltas)["passed"]


def test_downstream_dependency_audit_rejects_stale_string_tool_argument() -> None:
    events = [
        {"event_id": "c1", "role": "assistant", "tool_call": {"name": "write", "arguments": {}}},
        {"event_id": "r1", "role": "tool", "call_event_id": "c1", "tool_result": {"content": "{}", "error": False}},
        {"event_id": "c2", "role": "assistant", "tool_call": {"name": "follow_up", "arguments": {"item_id": "old_item_123"}}},
    ]
    deltas = [{
        "event_id": "r1",
        "baseline_result": {"content": {"item_id": "old_item_123"}, "error": False},
        "mutated_result": {"content": {"item_id": "new_item_456"}, "error": False},
    }]
    audit = downstream_dependency_audit(events, "c1", deltas)
    assert not audit["passed"]
    assert audit["issues"][0]["stale_baseline_strings"] == ["old_item_123"]
    events[-1]["tool_call"]["arguments"]["item_id"] = "new_item_456"
    assert downstream_dependency_audit(events, "c1", deltas)["passed"]


def test_downstream_dependency_audit_ignores_repeated_common_enum() -> None:
    events = [
        {"event_id": "c1", "role": "assistant", "tool_call": {"name": "write", "arguments": {}}},
        {"event_id": "r1", "role": "tool", "call_event_id": "c1", "tool_result": {"content": "{}", "error": False}},
        {"event_id": "c2", "role": "assistant", "tool_call": {"name": "other_write", "arguments": {"cabin": "economy"}}},
    ]
    deltas = [{
        "event_id": "r1",
        "baseline_result": {"content": {"cabin": "economy"}, "error": False},
        "mutated_result": {"content": {"cabin": "basic_economy"}, "error": False},
    }]
    assert downstream_dependency_audit(events, "c1", deltas)["passed"]


def semantic_fixture() -> tuple[dict, list[dict], dict]:
    trajectory = fixture_trajectory()
    trajectory["events"].append({"event_id": "a2", "role": "assistant", "content": "Done — the booking total is $384."})
    deltas = [{
        "event_id": "r1",
        "baseline_result": {"error": False, "content": {"total": 255}},
        "mutated_result": {"error": False, "content": {"total": 384}},
    }]
    verdict = {
        "overall_verdict": "consistent",
        "delta_checks": [{
            "delta_event_id": "r1",
            "verdict": "consistent",
            "changed_fact_summary": "The total changed from $255 to $384.",
            "evidence": [{"claim_id": "a2#L1"}],
            "explanation": "The final message matches the mutated result.",
        }],
        "unsupported_claims": [],
        "reason": "All changed facts are consistent.",
    }
    return trajectory, deltas, verdict


def test_semantic_verdict_requires_complete_exact_grounded_evidence() -> None:
    trajectory, deltas, verdict = semantic_fixture()
    audit = validate_semantic_verdict(trajectory, "c1", deltas, verdict, "semantic_verifier")
    assert audit["passed"], audit
    assert audit["checked_evidence_claims"] == 1


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda value: value.update(delta_checks=[]), "missed causal deltas"),
        (lambda value: value["delta_checks"][0].update(delta_event_id="invented"), "invented causal deltas"),
        (lambda value: value["delta_checks"][0]["evidence"][0].update(claim_id="invented#L99"), "claim ID is not valid"),
        (lambda value: value["delta_checks"][0].update(verdict="contradiction"), "semantic verdict is contradiction"),
        (lambda value: value.update(overall_verdict="uncertain"), "overall verdict is 'uncertain'"),
        (lambda value: value.update(unsupported_claims=[{"claim_id": "a2#L1", "explanation": "unsupported"}]), "unsupported claim"),
    ],
)
def test_semantic_verdict_rejects_adversarial_failures(mutation, expected_error: str) -> None:
    trajectory, deltas, verdict = semantic_fixture()
    mutation(verdict)
    audit = validate_semantic_verdict(trajectory, "c1", deltas, verdict, "semantic_falsifier")
    assert not audit["passed"]
    assert any(expected_error in error for error in audit["errors"]), audit


def test_semantic_not_mentioned_requires_no_evidence() -> None:
    trajectory, deltas, verdict = semantic_fixture()
    trajectory["events"][-1]["content"] = "Done — the booking was created."
    verdict["delta_checks"][0]["verdict"] = "not_mentioned"
    verdict["delta_checks"][0]["evidence"] = []
    assert validate_semantic_verdict(trajectory, "c1", deltas, verdict, "semantic_verifier")["passed"]
    verdict["delta_checks"][0]["evidence"] = [{"claim_id": "a2#L1"}]
    audit = validate_semantic_verdict(trajectory, "c1", deltas, verdict, "semantic_verifier")
    assert not audit["passed"]
    assert any("must not cite evidence" in error for error in audit["errors"])


def test_semantic_verdict_rejects_empty_reasoning_fields() -> None:
    trajectory, deltas, verdict = semantic_fixture()
    verdict["reason"] = ""
    verdict["delta_checks"][0]["changed_fact_summary"] = ""
    verdict["delta_checks"][0]["explanation"] = ""
    audit = validate_semantic_verdict(trajectory, "c1", deltas, verdict, "semantic_verifier")
    assert not audit["passed"]
    assert any("reason is empty" in error for error in audit["errors"])
    assert any("changed_fact_summary is empty" in error for error in audit["errors"])
    assert any("explanation is empty" in error for error in audit["errors"])


def test_semantic_evidence_cannot_point_before_target() -> None:
    trajectory, deltas, verdict = semantic_fixture()
    verdict["delta_checks"][0]["evidence"] = [{"claim_id": "a1#L1"}]
    audit = validate_semantic_verdict(trajectory, "c1", deltas, verdict, "semantic_verifier")
    assert not audit["passed"]
    assert any("claim ID is not valid later assistant prose" in error for error in audit["errors"])


def test_assistant_claim_units_are_deterministic_and_line_addressable() -> None:
    trajectory = fixture_trajectory()
    trajectory["events"].append({"event_id": "a2", "role": "assistant", "content": "First fact.\n\nSecond fact."})
    assert assistant_claim_units_after_target(trajectory, "c1") == [
        {"claim_id": "a2#L1", "event_id": "a2", "text": "First fact."},
        {"claim_id": "a2#L3", "event_id": "a2", "text": "Second fact."},
    ]


def test_tool_argument_enum_constraints_reject_humanized_value() -> None:
    trajectory = fixture_trajectory()
    trajectory["events"][3]["tool_call"] = {
        "name": "update_reservation_flights",
        "arguments": {"cabin": "basic economy"},
    }
    errors = validate_tool_argument_constraints(trajectory, ["c1"])
    assert any("basic_economy" in error and "basic economy" in error for error in errors)
    trajectory["events"][3]["tool_call"]["arguments"]["cabin"] = "basic_economy"
    assert validate_tool_argument_constraints(trajectory, ["c1"]) == []


def test_luna_retries_identical_request_when_response_has_no_output(monkeypatch) -> None:
    responses = [
        {"usage": {"input_tokens": 10, "output_tokens": 1}},
        {"output_text": "{}", "usage": {"input_tokens": 10, "output_tokens": 2}},
    ]
    calls: list[dict] = []

    def fake_post_json(url, body, api_key, timeout_s, max_retries):
        calls.append(body)
        return responses.pop(0)

    monkeypatch.setattr(augmentation_module, "post_json", fake_post_json)
    monkeypatch.setattr(augmentation_module, "env_value", lambda name: "test-key")
    monkeypatch.setattr(augmentation_module.time, "sleep", lambda seconds: None)
    result = call_luna(
        "semantic_verifier",
        {"trajectory": fixture_trajectory(), "target_event_id": "c1", "causal_deltas": []},
        max_retries=1,
    )
    assert result["output_text"] == "{}"
    assert result["api_attempt_count"] == 2
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 3, "cached_input_tokens": 0}
    assert calls[0] == calls[1]


def test_luna_rejects_unexpected_served_model(monkeypatch) -> None:
    monkeypatch.setattr(
        augmentation_module,
        "post_json",
        lambda *args, **kwargs: {"model": "unexpected-model", "output_text": "{}", "usage": {}},
    )
    monkeypatch.setattr(augmentation_module, "env_value", lambda name: "test-key")
    with pytest.raises(augmentation_module.ProviderError, match="unexpected served model"):
        call_luna(
            "semantic_verifier",
            {"trajectory": fixture_trajectory(), "target_event_id": "c1", "causal_deltas": []},
            max_retries=0,
        )


def test_semantic_verifier_roles_must_be_distinct_and_complete() -> None:
    assert semantic_verifier_stages({"semantic_verifier_stages": ["semantic_verifier", "semantic_falsifier"]}) == [
        "semantic_verifier",
        "semantic_falsifier",
    ]
    with pytest.raises(ValueError, match="duplicates"):
        semantic_verifier_stages({"semantic_verifier_stages": ["semantic_verifier", "semantic_verifier"]})
    with pytest.raises(ValueError, match="must contain"):
        semantic_verifier_stages({"semantic_verifier_stages": ["semantic_verifier"]})


def test_request_key_changes_when_pipeline_code_changes(monkeypatch) -> None:
    packet = {"trajectory": fixture_trajectory(), "changed_event_ids": ["c1"]}
    monkeypatch.setattr(augmentation_module, "pipeline_code_hash", lambda: "code-a")
    first = request_key("reconciler", packet)
    monkeypatch.setattr(augmentation_module, "pipeline_code_hash", lambda: "code-b")
    second = request_key("reconciler", packet)
    assert first != second


def test_spend_cap_uses_conservative_pre_call_bound() -> None:
    packet = {"trajectory": fixture_trajectory(), "target_event_id": "c1", "causal_deltas": []}
    bound = luna_call_cost_upper_bound("semantic_verifier", packet)
    assert bound > 0
    with pytest.raises(RuntimeError, match="would be exceeded"):
        enforce_call_spend_cap(0.0, bound / 2, "semantic_verifier", packet, None, call_luna)
    enforce_call_spend_cap(0.0, bound + 0.01, "semantic_verifier", packet, None, call_luna)


def test_run_pilot_automatically_retries_rejected_generation(tmp_path: Path, monkeypatch) -> None:
    trajectory = fixture_trajectory()
    seen_packets: list[dict] = []

    monkeypatch.setattr(
        augmentation_module,
        "select_pilot_seeds",
        lambda local_root: [{"domain": "airline", "trajectory": trajectory, "write_event_ids": ["c1"]}],
    )
    monkeypatch.setattr(augmentation_module, "build_packet", lambda local_root, value: fixture_packet(value))

    def fake_call(stage: str, packet: dict, plan: dict | None = None) -> dict:
        seen_packets.append(copy.deepcopy(packet))
        if stage == "planner":
            output = valid_plan()
            if packet.get("generation_attempt") is None:
                output["target_rule_id"] = "invalid.rule"
        elif stage == "editor":
            output = {
                "decision": "apply",
                "patches": [{
                    "operation": "replace_content",
                    "event_id": "a1",
                    "new_content": "Please confirm the booking.",
                    "new_arguments_json": None,
                    "event_ids": [],
                    "new_order": [],
                    "reason": "test retry",
                }],
                "requires_environment_replay": False,
                "changed_event_ids": ["a1"],
                "violation_explanation": "The amount is no longer explicit.",
            }
        else:
            raise AssertionError(stage)
        return {"output_text": json.dumps(output), "estimated_cost_usd": 0.0}

    result = run_pilot(tmp_path / "local", tmp_path / "outputs", call_fn=fake_call)
    assert result["passed"], result
    rows = [json.loads(line) for line in (tmp_path / "outputs" / "augmentation_pilot" / "pilot_results.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == ["planner_invalid", "ready_for_human_review"]
    assert rows[-1]["generation_attempt"] == 1
    retry_packets = [packet for packet in seen_packets if packet.get("generation_attempt") == 1]
    assert retry_packets
    assert retry_packets[0]["prior_rejection"]["previous_status"] == "planner_invalid"


def test_replay_validation_requires_successful_state_change_and_preserves_links() -> None:
    trajectory = fixture_trajectory()
    plan = valid_plan()
    replayed = {
        "events": trajectory["events"],
        "replay": {
            "target_result": {"content": "ok", "error": False},
            "target_state_changed": True,
            "final_state_hash": "abc",
            "baseline_source_fidelity": {"passed": True, "errors": []},
            "downstream_dependency_audit": {"passed": True, "errors": [], "issues": []},
        },
    }
    audit = validate_replayed_trajectory(trajectory, replayed, plan)
    assert audit["passed"], audit
    replayed["events"] = copy.deepcopy(replayed["events"])
    replayed["events"][-1]["tool_result"] = None
    audit = validate_replayed_trajectory(trajectory, replayed, plan)
    assert not audit["passed"]
    assert "replay lost tool result at r1" in audit["errors"]
