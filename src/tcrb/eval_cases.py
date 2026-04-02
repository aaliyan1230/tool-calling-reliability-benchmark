from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_eval_cases(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("eval cases file must be a JSON object")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("eval cases file must contain non-empty 'cases' list")
    return payload


def score_eval_cases(result_payload: dict[str, Any], eval_cases_payload: dict[str, Any]) -> dict[str, Any]:
    cases = list(eval_cases_payload.get("cases", []))
    case_map = {str(case.get("task_id", "")): case for case in cases}

    policy_groups: dict[str, list[dict[str, Any]]] = {}
    for row in result_payload.get("task_results", []):
        task_id = str(row.get("task_id", ""))
        if task_id not in case_map:
            continue
        policy = str(row.get("policy", ""))
        policy_groups.setdefault(policy, []).append(row)

    policy_scores: list[dict[str, Any]] = []
    for policy in sorted(policy_groups.keys()):
        rows = policy_groups[policy]
        first_matches = 0
        seq_matches = 0
        evaluated = 0
        mismatches: list[dict[str, Any]] = []

        for row in rows:
            task_id = str(row.get("task_id", ""))
            case = case_map.get(task_id)
            if case is None:
                continue

            attempts = list(row.get("attempts", []))
            called = [str(a.get("tool_name", "")) for a in attempts]
            called_first = called[0] if called else ""

            expected_first = str(case.get("expected_first_tool", ""))
            expected_seq = [str(name) for name in case.get("expected_tool_sequence", []) if str(name)]

            first_ok = bool(expected_first) and called_first == expected_first
            seq_ok = bool(expected_seq) and called[: len(expected_seq)] == expected_seq

            evaluated += 1
            if first_ok:
                first_matches += 1
            if seq_ok:
                seq_matches += 1
            if not (first_ok and seq_ok):
                mismatches.append(
                    {
                        "task_id": task_id,
                        "question": case.get("question"),
                        "expected_first_tool": expected_first,
                        "called_first_tool": called_first,
                        "expected_tool_sequence": expected_seq,
                        "called_prefix": called[: len(expected_seq)] if expected_seq else called[:1],
                    }
                )

        policy_scores.append(
            {
                "policy": policy,
                "cases_evaluated": evaluated,
                "first_tool_accuracy": (first_matches / evaluated) if evaluated else 0.0,
                "sequence_prefix_accuracy": (seq_matches / evaluated) if evaluated else 0.0,
                "mismatches": mismatches,
            }
        )

    return {
        "toolset_id": eval_cases_payload.get("toolset_id"),
        "cases_total": len(cases),
        "policies": policy_scores,
    }


def write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
