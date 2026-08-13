"""Replay one augmented trace against the official tau2 environment.

This file is intentionally a small JSON-in/JSON-out adapter.  It runs in the
tau2 v1.0.1 virtualenv because the benchmark repository does not depend on
tau2 at runtime.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _content(value: Any) -> str:
    value = _jsonable(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _semantic(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _recorded_result(events: list[dict[str, Any]], call: dict[str, Any]) -> dict[str, Any] | None:
    call_id = call.get("tool_call_id")
    event_id = call.get("event_id")
    for event in events:
        if event.get("role") != "tool":
            continue
        if event.get("call_event_id") == event_id or (call_id and event.get("tool_call_id") == call_id):
            result = event.get("tool_result")
            if isinstance(result, dict):
                return result
            if isinstance(event.get("content"), str):
                return {"content": event["content"], "error": False}
    return None


def _replace_result_events(events: list[dict[str, Any]], call: dict[str, Any], result: dict[str, Any]) -> None:
    call_id = call.get("tool_call_id")
    event_id = call.get("event_id")
    for event in events:
        if event.get("role") != "tool":
            continue
        if event.get("call_event_id") == event_id or (call_id and event.get("tool_call_id") == call_id):
            event["content"] = result["content"]
            event["tool_result"] = copy.deepcopy(result)


def _state_hash(env: Any) -> str:
    import hashlib

    db = None
    for toolkit in (getattr(env, "tools", None), getattr(env, "user_tools", None)):
        if toolkit is not None and getattr(toolkit, "db", None) is not None:
            db = toolkit.db
            break
    if db is None:
        return "no-db"
    payload = json.dumps(_jsonable(db), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _make_environment(domain: str):
    if domain == "airline":
        from tau2.domains.airline.environment import get_environment
    elif domain == "retail":
        from tau2.domains.retail.environment import get_environment
    else:
        raise ValueError(f"replay is not configured for domain {domain!r}")
    return get_environment()


def _apply_task_initial_state(env: Any, domain: str, task_id: str) -> None:
    """Apply task setup when present; current airline/retail seeds have none."""
    if domain == "airline":
        from tau2.domains.airline.environment import get_tasks
    else:
        from tau2.domains.retail.environment import get_tasks
    task = next((item for item in get_tasks(None) if str(item.id) == str(task_id)), None)
    if task is None:
        raise ValueError(f"task {task_id!r} not found in tau2 data")
    initial = task.initial_state
    if initial is None:
        return
    if initial.initialization_data is not None:
        # A data patch changes the starting world and cannot be safely guessed.
        raise ValueError("task initialization_data replay is not supported by this adapter")
    for action in initial.initialization_actions or []:
        env.run_env_function_call(action)


def replay(payload: dict[str, Any]) -> dict[str, Any]:
    domain = str(payload["domain"])
    task_id = str(payload["task_id"])
    events = copy.deepcopy(payload["events"])
    mutations = payload.get("mutations") or {}
    env = _make_environment(domain)
    _apply_task_initial_state(env, domain, task_id)
    before_target_hash: str | None = None
    target_event_id = payload.get("target_event_id")
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    target_result: dict[str, Any] | None = None
    target_state_changed = False
    target_state_hash_before: str | None = None
    target_state_hash_after: str | None = None
    for event in events:
        call = event.get("tool_call")
        if not isinstance(call, dict):
            continue
        event_id = event.get("event_id")
        if event_id == target_event_id:
            before_target_hash = _state_hash(env)
            target_state_hash_before = before_target_hash
        args = copy.deepcopy((mutations.get(event_id) or {}).get("arguments", call.get("arguments", {})))
        if event_id in mutations:
            # The replayed trace must show the actual changed call as well as
            # the regenerated result events.
            event["tool_call"]["arguments"] = copy.deepcopy(args)
        before = _state_hash(env)
        try:
            value = env.make_tool_call(call["name"], requestor="assistant", **args)
            result = {"content": _content(value), "error": False}
        except Exception as exc:  # tau2 encodes tool failures as error results
            result = {"content": str(exc), "error": True}
        after = _state_hash(env)
        _replace_result_events(events, event, result)
        recorded = _recorded_result(payload["events"], event)
        prefix_match = True
        if event_id != target_event_id and before_target_hash is None and recorded is not None:
            prefix_match = result.get("error") == recorded.get("error") and _semantic(result.get("content")) == _semantic(recorded.get("content"))
            if not prefix_match:
                errors.append(f"prefix result mismatch at {event_id}")
        if event_id == target_event_id:
            target_result = result
            target_state_changed = before != after
            target_state_hash_after = after
        calls.append({
            "event_id": event_id,
            "tool": call.get("name"),
            "error": result["error"],
            "state_changed": before != after,
            "prefix_match": prefix_match,
        })
    if target_event_id is None:
        errors.append("no target_event_id supplied")
    elif target_result is None:
        errors.append(f"target event not found: {target_event_id}")
    elif target_result.get("error"):
        errors.append("mutated target tool call failed")
    elif not target_state_changed:
        errors.append("mutated target tool call did not change database state")
    return {
        "passed": not errors,
        "errors": errors,
        "domain": domain,
        "task_id": task_id,
        "target_event_id": target_event_id,
        "target_result": target_result,
        "target_state_changed": target_state_changed,
        "target_state_hash_before": target_state_hash_before,
        "target_state_hash_after": target_state_hash_after,
        "calls": calls,
        "events": events,
        "final_state_hash": _state_hash(env),
    }


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = replay(payload)
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
