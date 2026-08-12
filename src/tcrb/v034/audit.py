from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .schema import SUMMARY_KEYS, validate_summary
from .sources import audit_sources
from .summaries import FORBIDDEN_INPUT_KEYS, source_input
from .util import DEFAULT_LOCAL_ROOT, DEFAULT_RUN_ROOT, read_json, read_jsonl, sha256_file, write_json


def validate_outputs(local_root: Path = DEFAULT_LOCAL_ROOT, run_root: Path = DEFAULT_RUN_ROOT, stage: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    normalized = local_root / "normalized"
    trajectories = read_jsonl(normalized / "trajectories.jsonl")
    ids = [row.get("trajectory_id") for row in trajectories]
    if len(ids) != len(set(ids)):
        errors.append("duplicate trajectory IDs")
    for trajectory in trajectories:
        event_ids = [event.get("event_id") for event in trajectory.get("events", [])]
        if len(event_ids) != len(set(event_ids)):
            errors.append(f"duplicate event ID in {trajectory.get('trajectory_id')}")
        input_value = source_input(trajectory)
        if nested_keys(input_value) & FORBIDDEN_INPUT_KEYS:
            errors.append(f"private field leaked into summary input {trajectory.get('trajectory_id')}")
    frozen = local_root / "frozen_manifest.json"
    if frozen.exists():
        manifest = read_json(frozen)
        if manifest.get("final_pairs") != 30 or manifest.get("by_domain") != {"airline": 15, "retail": 15}:
            errors.append("frozen dataset is not exactly 15 airline + 15 retail pairs")
    if stage:
        run_dir = run_root / stage
        responses = read_jsonl(run_dir / "summary_responses.jsonl")
        for row in responses:
            if row.get("status") != "success":
                continue
            trajectory = next((item for item in trajectories if item["trajectory_id"] == row.get("trajectory_id")), None)
            if trajectory is None:
                errors.append(f"summary references missing trajectory {row.get('trajectory_id')}")
                continue
            errors.extend(f"{row.get('call_id')}: {error}" for error in validate_summary(row.get("summary"), {event["event_id"] for event in trajectory["events"]}))
            if row.get("validation_errors"):
                errors.extend(f"{row.get('call_id')}: {error}" for error in row["validation_errors"])
        if not responses:
            errors.append(f"no summary responses in {run_dir}")
    result = {"version": "v034-output-audit-1", "passed": not errors, "errors": errors, "trajectory_count": len(trajectories), "stage": stage}
    write_json((run_root / (stage or "dataset")) / "validation.json", result)
    return result


def audit_run(local_root: Path = DEFAULT_LOCAL_ROOT, run_root: Path = DEFAULT_RUN_ROOT, stage: str | None = None) -> dict[str, Any]:
    source_audit = read_json(local_root / "source_audit.json") if (local_root / "source_audit.json").exists() else {"passed": False, "errors": ["source audit has not been run"]}
    output = validate_outputs(local_root, run_root, stage)
    artifacts: dict[str, str] = {}
    for root in (local_root, run_root / stage if stage else local_root):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name not in {"audit.json", "validation.json"}:
                artifacts[str(path)] = sha256_file(path)
    result = {"version": "v034-audit-1", "passed": bool(source_audit.get("passed")) and bool(output.get("passed")), "source_passed": bool(source_audit.get("passed")), "output_passed": bool(output.get("passed")), "errors": list(source_audit.get("errors", [])) + list(output.get("errors", [])), "artifact_sha256": artifacts}
    write_json(run_root / (stage or "dataset") / "audit.json", result)
    return result


def nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(nested_keys(item) for item in value)) if value else set()
    return set()
