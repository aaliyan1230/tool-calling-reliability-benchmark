from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tcrb.v034 import augmentation
from tcrb.v034.hard import (
    DEFAULT_LOCAL_ROOT,
    HARD_SEED_REGISTRY_NAME,
    HARD_VERSION,
    hard_case_validation,
    hard_config_context,
    load_hard_config,
    make_hard_review_packet,
    repair_hard_plan_intervening_ids,
    select_hard_seeds,
)


def hard_fixture() -> tuple[dict, dict, dict, dict, dict, list[str]]:
    original = {
        "trajectory_id": "traj_hard_fixture",
        "domain": "airline",
        "task_id": "999",
        "events": [
            {"event_id": "u1", "role": "user", "content": "Please cancel the booking if it is eligible. Yes."},
            {"event_id": "a1", "role": "assistant", "content": "I will check the booking and cancellation conditions."},
            {"event_id": "c1", "role": "assistant", "tool_call": {"name": "book_reservation", "arguments": {"x": 1}}},
            {"event_id": "r1", "role": "tool", "call_event_id": "c1", "tool_result": {"content": "{\"ok\":true}", "error": False}},
            {"event_id": "a2", "role": "assistant", "content": "The booking details changed while I checked."},
            {"event_id": "c2", "role": "assistant", "tool_call": {"name": "cancel_reservation", "arguments": {"reservation_id": "r1"}}},
            {"event_id": "r2", "role": "tool", "call_event_id": "c2", "tool_result": {"content": "{\"ok\":true}", "error": False}},
        ],
    }
    augmented = copy.deepcopy(original)
    augmented["events"][5]["tool_call"]["arguments"]["reservation_id"] = "r2"
    plan = {
        "decision": "mutate",
        "selected_family": "stale_confirmation",
        "target_rule_id": "airline.cancel.eligibility",
        "target_write_event_id": "c2",
        "supporting_event_ids": ["u1"],
        "intervening_event_ids": ["a1", "c1", "r1", "a2"],
        "target_write_rank": 2,
        "procedural_timeline": "approval -> state check -> changed booking -> stale cancellation",
        "precondition_summary": "The cancellation eligibility fact changed before the write.",
        "mutation_summary": "Keep the old cancellation action after the booking changed.",
    }
    editor = {"requires_environment_replay": True}
    packet = {
        "hard_case_spec": {
            "case_id": "fixture",
            "family": "stale_confirmation",
            "allowed_target_rules": ["airline.cancel.eligibility"],
            "required_intervening_events": 2,
        }
    }
    return original, augmented, plan, editor, packet, ["c2"]


@pytest.mark.local_data
def test_hard_registry_has_locked_quota_and_no_v034_overlap() -> None:
    smoke = select_hard_seeds(stage="smoke")
    assert len(smoke) == 4
    assert {row["domain"] for row in smoke} == {"airline", "retail"}
    if (DEFAULT_LOCAL_ROOT / HARD_SEED_REGISTRY_NAME).exists():
        core = select_hard_seeds(stage="core")
        assert len(core) == 16
        assert {row["domain"] for row in core} == {"airline", "retail"}
    else:
        with pytest.raises(ValueError, match="family needs at least"):
            select_hard_seeds(stage="core")


def test_hard_planner_schema_requires_timeline_fields() -> None:
    with hard_config_context():
        schema = augmentation.planner_json_schema()
        assert {"intervening_event_ids", "target_write_rank", "procedural_timeline", "precondition_summary"}.issubset(
            set(schema["required"])
        )
        prompt = augmentation.prompt_text("planner")
    assert "immediately-after-confirmation" in prompt
    assert "real production agent" in prompt
    assert schema["properties"]["target_write_rank"]["minimum"] == 1
    assert "grounded in facts already present in the trace" in prompt
    assert "shorthand such as" in prompt


def test_hard_planner_user_includes_family_specific_thresholds() -> None:
    packet = {
        "trajectory": {"trajectory_id": "traj", "domain": "airline", "events": []},
        "policy": "policy",
        "policy_rules": [],
        "hard_case_spec": {
            "family": "stale_confirmation",
            "min_source_writes": 1,
            "min_target_write_rank": 1,
            "required_intervening_events": 2,
            "allowed_target_rules": ["airline.cancel.eligibility"],
        },
    }
    with hard_config_context():
        user = augmentation.build_planner_user(packet)
    assert "at least 1 source write(s)" in user
    assert "target write rank 1 or higher" in user
    assert "2 meaningful intervening event(s)" in user
    assert '"decision":"mutate" or "not_applicable"' in user


def test_hard_plan_repair_only_fills_deterministic_gap_ids() -> None:
    original, _, plan, _, _, _ = hard_fixture()
    plan["intervening_event_ids"] = ["a1"]
    added = repair_hard_plan_intervening_ids(plan, original)
    assert added == ["c1", "r1", "a2"]
    assert plan["intervening_event_ids"] == ["a1", "c1", "r1", "a2"]


def test_hard_plan_repair_does_not_touch_not_applicable() -> None:
    original, _, plan, _, _, _ = hard_fixture()
    plan["decision"] = "not_applicable"
    plan["intervening_event_ids"] = []
    assert repair_hard_plan_intervening_ids(plan, original) == []
    assert plan["intervening_event_ids"] == []


def test_hard_review_packet_reads_hard_stage_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tcrb.v034.hard as hard_module

    seed = {"domain": "airline", "trajectory": {"trajectory_id": "traj_review", "domain": "airline"}}
    monkeypatch.setattr(hard_module, "select_hard_seeds", lambda local_root, stage, run_root=None: [seed])
    monkeypatch.setattr(hard_module, "_case_rows", lambda stage, local_root: [seed])
    monkeypatch.setattr(hard_module, "hard_seed_set_hash", lambda stage, local_root, run_root=None: "cfg")
    monkeypatch.setattr("tcrb.v034.augmentation.policy_context", lambda local_root, domain: {"policy": "policy", "policy_rules": []})
    run_dir = tmp_path / "augmentation_hard_smoke"
    run_dir.mkdir(parents=True)
    (run_dir / "pilot_results.jsonl").write_text(
        json.dumps({
            "trajectory_id": "traj_review",
            "config_hash": "cfg",
            "status": "ready_for_human_review",
            "augmented_trajectory_id": "aug_review",
            "trajectory": {"trajectory_id": "traj_review", "domain": "airline", "events": []},
            "augmented_trajectory": {"trajectory_id": "aug_review", "domain": "airline", "events": []},
        }) + "\n",
        encoding="utf-8",
    )
    result = make_hard_review_packet(tmp_path, tmp_path, "smoke")
    assert result["rows"] == 1
    assert result["passed"]
    assert "augmentation_hard_smoke" in result["packet_path"]


def test_hard_config_is_deepseek_only_at_minimal_reasoning() -> None:
    config = load_hard_config()
    assert config["provider"] == "deepseek"
    assert config["model"] == "deepseek-v4-flash"
    assert config["reasoning_effort"] == "none"


def test_hard_case_validation_accepts_a_non_adjacent_procedural_candidate() -> None:
    original, augmented, plan, editor, packet, changed = hard_fixture()
    result = hard_case_validation(original, augmented, plan, editor, changed, packet)
    assert result["passed"], result
    assert result["target_write_rank"] == 2
    assert result["write_count"] == 2


def test_hard_case_validation_allows_one_write_when_the_gap_is_real() -> None:
    original = {
        "trajectory_id": "traj_one_write",
        "domain": "airline",
        "task_id": "998",
        "events": [
            {"event_id": "u1", "role": "user", "content": "Please cancel if eligible."},
            {"event_id": "a1", "role": "assistant", "content": "I checked the booking."},
            {"event_id": "a2", "role": "assistant", "content": "The booking state changed while checking."},
            {"event_id": "c1", "role": "assistant", "tool_call": {"name": "cancel_reservation", "arguments": {"reservation_id": "old"}}},
            {"event_id": "r1", "role": "tool", "call_event_id": "c1", "tool_result": {"content": "{\"ok\":true}", "error": False}},
        ],
    }
    augmented = copy.deepcopy(original)
    augmented["events"][3]["tool_call"]["arguments"]["reservation_id"] = "new"
    plan = {
        "decision": "mutate",
        "selected_family": "stale_confirmation",
        "target_rule_id": "airline.cancel.eligibility",
        "target_write_event_id": "c1",
        "supporting_event_ids": ["u1"],
        "intervening_event_ids": ["a1", "a2"],
        "target_write_rank": 1,
    }
    packet = {"hard_case_spec": {
        "case_id": "one-write",
        "family": "stale_confirmation",
        "allowed_target_rules": ["airline.cancel.eligibility"],
        "required_intervening_events": 2,
        "min_source_writes": 1,
        "min_target_write_rank": 1,
    }}
    result = hard_case_validation(original, augmented, plan, {"requires_environment_replay": True}, ["c1"], packet)
    assert result["passed"], result
    assert result["target_write_rank"] == 1
    assert result["write_count"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "wrong_family",
        "adjacent",
        "user_edit",
        "unchanged_target",
    ],
)
def test_hard_case_validation_rejects_easy_or_tampered_candidates(change: str) -> None:
    original, augmented, plan, editor, packet, changed = hard_fixture()
    if change == "wrong_family":
        plan["selected_family"] = "partial_or_bundled_confirmation"
    elif change == "adjacent":
        plan["supporting_event_ids"] = ["a2"]
        plan["intervening_event_ids"] = []
    elif change == "user_edit":
        augmented["events"][0]["content"] = "Yes, cancel it."
    elif change == "unchanged_target":
        changed = ["a2"]
    result = hard_case_validation(original, augmented, plan, editor, changed, packet)
    assert not result["passed"]
    assert result["errors"]


@pytest.mark.local_data
def test_hard_smoke_full_runner_with_fake_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the resumable full path without making paid API or tau2 calls."""
    def fake_replay(original, augmented, plan, run_dir):
        target = plan["target_write_event_id"]
        delta = {
            "event_id": target,
            "baseline_result": {"error": False, "content": {"state": "old"}},
            "mutated_result": {"error": False, "content": {"state": "new"}},
        }
        return {
            "passed": True,
            "events": copy.deepcopy(augmented["events"]),
            "baseline_events": copy.deepcopy(original["events"]),
            "target_state_hash_before": "same",
            "baseline_target_state_hash_before": "same",
            "target_result": {"error": False, "content": {"ok": True}},
            "target_state_changed": True,
            "final_state_hash": "mutated",
            "baseline_final_state_hash": "baseline",
            "causal_changed_event_ids": [target],
            "causal_deltas": [delta],
            "pre_target_differences": [],
            "differential_replay": {"passed": True, "errors": [], "pre_target_differences": []},
            "downstream_dependency_audit": {"passed": True, "errors": []},
            "baseline_source_fidelity": {"passed": True, "errors": []},
        }

    def fake_call(stage: str, packet: dict, plan: dict | None = None) -> dict:
        events = packet["trajectory"]["events"]
        if stage == "planner":
            writes = packet["write_event_ids_for_validation_only"]
            target = writes[1]
            positions = {event["event_id"]: index for index, event in enumerate(events)}
            support = [event["event_id"] for event in events[:positions[target]] if event.get("role") == "user"][:1]
            support = support or [events[0]["event_id"]]
            gap = [
                event["event_id"]
                for event in events[positions[support[-1]] + 1 : positions[target]]
                if event.get("content") or event.get("tool_call")
            ]
            output = {
                "decision": "mutate",
                "selected_family": packet["hard_case_spec"]["family"],
                "target_rule_id": packet["hard_case_spec"]["allowed_target_rules"][0],
                "target_write_event_id": target,
                "supporting_event_ids": support,
                "intervening_event_ids": gap,
                "target_write_rank": 2,
                "procedural_timeline": "approval -> intervening state evidence -> later write",
                "precondition_summary": "Later write uses stale procedural context.",
                "mutation_summary": "Change one later write.",
                "realism_reason": "Long workflows can retain stale state.",
                "subtlety_reason": "The failure requires linking multiple events.",
                "protected_facts": ["identity", "goal"],
                "proposed_changes": [{"event_id": target, "operation": "replace_tool_arguments", "field_or_scope": "arguments", "intent": "change target"}],
                "requires_environment_replay": True,
            }
        elif stage == "editor":
            target = plan["target_write_event_id"]
            event = next(event for event in events if event["event_id"] == target)
            arguments = copy.deepcopy((event.get("tool_call") or {}).get("arguments") or {})
            arguments["_hard_smoke_mutation"] = "different-but-valid"
            output = {
                "decision": "apply",
                "patches": [{"operation": "replace_tool_arguments", "event_id": target, "new_content": None, "new_arguments_json": json.dumps(arguments), "event_ids": [], "new_order": [], "reason": "one later write"}],
                "requires_environment_replay": True,
                "changed_event_ids": [target],
                "violation_explanation": "The target violates the selected procedural rule.",
            }
        elif stage == "reconciler":
            output = {"decision": "no_change", "patches": [], "changed_event_ids": [], "reason": "No prose changed."}
        else:
            target = packet["causal_deltas"][0]["event_id"]
            output = {"overall_verdict": "consistent", "delta_checks": [{"delta_event_id": target, "verdict": "not_mentioned", "changed_fact_summary": "The tool result changed but later prose does not state it.", "evidence": [], "explanation": "No later assistant claim covers the changed result."}], "unsupported_claims": [], "reason": "The delta is not mentioned."}
        return {"output_text": json.dumps(output), "estimated_cost_usd": 0.0}

    monkeypatch.setattr(augmentation, "replay_with_tau2", fake_replay)
    with hard_config_context():
        result = augmentation.run_pilot(
            Path("local/v034"),
            tmp_path / "outputs",
            call_fn=fake_call,
            seed_set="hard_smoke",
            spend_cap_usd=10,
        )
    assert result["passed"] is True, result
    assert result["ready_for_human_review"] == 4
    assert result["estimated_cost_usd"] == 0


def test_provider_failure_is_recorded_and_does_not_crash_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory = {
        "trajectory_id": "traj_provider_error",
        "domain": "airline",
        "task_id": "999",
        "events": [
            {"event_id": "u1", "role": "user", "content": "Please update it."},
            {"event_id": "c1", "role": "assistant", "tool_call": {"name": "book_reservation", "arguments": {}}},
        ],
    }
    monkeypatch.setattr(
        augmentation,
        "select_pilot_seeds",
        lambda local_root: [{"domain": "airline", "trajectory": trajectory, "write_event_ids": ["c1"]}],
    )
    monkeypatch.setattr(
        augmentation,
        "build_packet",
        lambda local_root, value: {
            "trajectory": {"trajectory_id": value["trajectory_id"], "domain": value["domain"], "events": value["events"]},
            "policy": "policy",
            "policy_rules": [],
            "state_changing_tool_contracts": {},
            "write_event_ids_for_validation_only": ["c1"],
        },
    )

    def failing_call(stage: str, packet: dict, plan: dict | None = None) -> dict:
        raise TimeoutError("provider timed out")

    result = augmentation.run_pilot(tmp_path / "local", tmp_path / "outputs", call_fn=failing_call)
    assert result["passed"] is False
    response_rows = [json.loads(line) for line in (tmp_path / "outputs" / "augmentation_pilot" / "llm_responses.jsonl").read_text().splitlines()]
    assert response_rows
    assert response_rows[0]["status"] == "provider_error"


def test_deepseek_transport_uses_same_hard_prompt_and_records_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_post(url, body, api_key, timeout_s, max_retries):
        calls.append({"url": url, "body": body, "api_key": api_key, "timeout_s": timeout_s, "max_retries": max_retries})
        return {
            "model": "deepseek-v4-flash",
            "choices": [{"message": {"content": '{"decision":"not_applicable"}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }

    monkeypatch.setattr(augmentation, "post_json", fake_post)
    monkeypatch.setattr(augmentation, "env_value", lambda *names: "test-key")
    with hard_config_context():
        result = augmentation.call_deepseek(
            "planner",
            {"trajectory": {"trajectory_id": "traj", "domain": "airline", "events": []}},
            timeout_s=7,
            max_retries=0,
        )
    assert result["provider"] == "opencode_go"
    assert result["model_id"] == "deepseek-v4-flash"
    assert result["output_text"] == '{"decision":"not_applicable"}'
    assert calls[0]["url"] == "https://opencode.ai/zen/go/v1/chat/completions"
    assert calls[0]["body"]["model"] == "deepseek-v4-flash"
    assert calls[0]["body"]["temperature"] == 0
    assert calls[0]["body"]["response_format"] == {"type": "json_object"}
    assert calls[0]["timeout_s"] == 7
