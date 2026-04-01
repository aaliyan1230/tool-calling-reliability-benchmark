from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..models import Workload


def _task_lookup_from_workload(workload: Workload | None) -> dict[str, dict[str, Any]]:
    if workload is None:
        return {}
    return {
        task.task_id: {
            "primary_tool": task.primary_tool,
            "fallback_tools": list(task.fallback_tools),
            "required_schema": list(task.required_schema),
        }
        for task in workload.tasks
    }


def build_examples_from_result_payload(
    payload: dict[str, Any],
    *,
    workload: Workload | None = None,
    include_failure_attempts: bool = False,
) -> list[dict[str, Any]]:
    task_lookup = _task_lookup_from_workload(workload)
    rows = payload.get("task_results", [])
    examples: list[dict[str, Any]] = []

    for row in rows:
        task_id = str(row.get("task_id", ""))
        policy = str(row.get("policy", ""))
        planner_id = str(row.get("planner_id", ""))
        attempts = list(row.get("attempts", []))
        attempted_before: list[str] = []
        last_status: str | None = None

        for attempt in attempts:
            status = str(attempt.get("status", ""))
            tool_name = str(attempt.get("tool_name", "")).strip()
            invalid_tool_call = bool(attempt.get("invalid_tool_call", False))

            keep = status == "success" or include_failure_attempts
            if not keep or invalid_tool_call or not tool_name:
                attempted_before.append(tool_name)
                last_status = status
                continue

            context = {
                "task_id": task_id,
                "policy": policy,
                "planner_id": planner_id,
                "attempt_number": int(attempt.get("attempt_number", 0)),
                "attempted_tools": [name for name in attempted_before if name],
                "last_status": last_status,
            }
            context.update(task_lookup.get(task_id, {}))

            examples.append(
                {
                    "prompt": context,
                    "completion": {"tool_name": tool_name},
                    "status": status,
                }
            )

            attempted_before.append(tool_name)
            last_status = status

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for example in examples:
        fingerprint = json.dumps(
            {
                "prompt": example["prompt"],
                "completion": example["completion"],
            },
            sort_keys=True,
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(example)

    return deduped


def split_examples(
    examples: list[dict[str, Any]],
    validation_split: float,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0.0, 1.0)")

    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    validation_size = int(round(len(shuffled) * validation_split))
    validation_size = min(max(validation_size, 0), len(shuffled))

    eval_rows = shuffled[:validation_size]
    train_rows = shuffled[validation_size:]
    return train_rows, eval_rows


def write_jsonl(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
