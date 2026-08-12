from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcrb.v034.audit import validate_outputs
from tcrb.v034.schema import validate_summary
from tcrb.v034.selection import cohens_kappa
from tcrb.v034.sources import normalize_task, validate_simulation
from tcrb.v034.summaries import action_receipt, call_matrix, parse_summary, summary_json_schema
from tcrb.v034.util import content_id, write_json, write_jsonl


def test_task_normalization_removes_only_schema_metadata() -> None:
    task = {"id": 1, "description": "x", "evaluation_criteria": {"env_assertions": None}, "ticket": {"secret": True}, "annotations": {"x": 1}}
    assert normalize_task(task) == {"id": 1, "description": "x", "evaluation_criteria": {}}


def test_simulation_validation_links_tool_results() -> None:
    simulation = {"messages": [{"role": "assistant", "tool_calls": [{"id": "c1", "name": "read", "arguments": {}}]}, {"role": "tool", "id": "c1", "content": "ok"}], "reward_info": {"db_check": {"db_match": True}}}
    assert validate_simulation(simulation) == []
    simulation["messages"][1]["id"] = "wrong"
    assert "tool result does not reference a call" in validate_simulation(simulation)


def test_summary_parser_and_citations() -> None:
    summary = {"user_request": "x", "key_facts": [{"text": "fact", "source_event_ids": ["e1"]}], "actions_and_results": [], "state_changes": [], "unresolved_or_risks": []}
    assert parse_summary("```json\n" + json.dumps(summary) + "\n```") == summary
    assert not validate_summary(summary, {"e1"})
    assert validate_summary(summary, set())


def test_receipt_is_deterministic_and_contains_write_result() -> None:
    trajectory = {"write_event_ids": ["c"], "events": [{"event_id": "c", "role": "assistant", "tool_call": {"name": "cancel_reservation", "arguments": {"id": "R1"}}}, {"event_id": "r", "role": "tool", "call_event_id": "c", "tool_result": {"content": "ok", "error": False}}]}
    assert action_receipt(trajectory) == action_receipt(trajectory)
    assert action_receipt(trajectory)["writes"][0]["result_event_id"] == "r"


def test_kappa() -> None:
    assert cohens_kappa(["safe", "unsafe"], ["safe", "unsafe"]) == 1.0
    assert cohens_kappa(["safe", "unsafe"], ["unsafe", "safe"]) < 0


def test_call_matrix_counts_for_fixture(tmp_path: Path) -> None:
    local = tmp_path / "local"
    run = tmp_path / "run"
    trajectories = []
    pairs = []
    for index in range(4):
        ids = []
        for role in ("safe", "unsafe"):
            tid = f"traj_{index}_{role}"
            ids.append(tid)
            trajectories.append({"trajectory_id": tid, "domain": "airline", "task_id": str(index), "source_agent": "gpt-5.2", "events": [{"event_id": f"{tid}_e", "role": "user", "content": "hello"}], "write_event_ids": []})
        pairs.append({"pair_id": f"p{index}", "domain": "airline", "task_id": str(index), "source_agent": "gpt-5.2", "safe_candidate_id": ids[0], "unsafe_candidate_id": ids[1]})
    write_jsonl(local / "normalized" / "trajectories.jsonl", trajectories)
    write_jsonl(local / "frozen_pairs_private.jsonl", pairs)
    write_jsonl(local / "dev_pairs_private.jsonl", pairs[:2])
    assert len(call_matrix(local, run, "smoke")) == 2 * 2 * 2 * 2
    assert len(call_matrix(local, run, "core")) == 4 * 2 * 2 * 2
    assert len(call_matrix(local, run, "stability")) == 8 * 2 * 2 * 3


def test_public_input_has_no_private_keys(tmp_path: Path) -> None:
    local = tmp_path / "local"
    run = tmp_path / "run"
    trajectory = {"trajectory_id": "t1", "domain": "airline", "task_id": "1", "source_agent": "gpt", "events": [{"event_id": "e1", "role": "user", "content": "hello"}], "write_event_ids": []}
    write_jsonl(local / "normalized" / "trajectories.jsonl", [trajectory])
    result = validate_outputs(local, run)
    assert result["passed"]
