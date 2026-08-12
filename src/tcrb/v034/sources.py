from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from .schema import WRITE_TOOLS
from .util import CONFIG_ROOT, DEFAULT_LOCAL_ROOT, canonical, content_id, read_json, sha256_file, write_json, write_jsonl


LOCK_PATH = CONFIG_ROOT / "source_lock.json"


def load_lock() -> dict[str, Any]:
    return read_json(LOCK_PATH)


def all_sources(lock: dict[str, Any], include_fallback: bool = True) -> list[dict[str, Any]]:
    groups = list(lock["primary"])
    if include_fallback:
        groups.extend(lock.get("fallback", []))
    return groups


def fetch_sources(local_root: Path = DEFAULT_LOCAL_ROOT, include_fallback: bool = True) -> dict[str, Any]:
    lock = load_lock()
    raw_root = local_root / "raw"
    fetched: list[dict[str, Any]] = []
    for source in all_sources(lock, include_fallback=include_fallback):
        for domain, meta in source["files"].items():
            target = raw_root / source["submission"] / meta["name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha256_file(target) != meta["sha256"]:
                url = f"{lock['bucket']}/{source['submission']}/trajectories/{meta['name']}"
                request = urllib.request.Request(url, headers={"User-Agent": "tcrb-v034/1.0"})
                with urllib.request.urlopen(request, timeout=180) as response, target.open("wb") as output:
                    shutil.copyfileobj(response, output)
            fetched.append({"agent": source["agent"], "submission": source["submission"], "domain": domain, "path": str(target), "sha256": sha256_file(target), "bytes": target.stat().st_size})
    write_json(local_root / "fetch_manifest.json", {"tau_version": lock["tau_version"], "files": fetched})
    return {"files": len(fetched), "root": str(raw_root), "include_fallback": include_fallback}


def audit_sources(local_root: Path = DEFAULT_LOCAL_ROOT, tau_root: Path | None = None, include_fallback: bool = True) -> dict[str, Any]:
    lock = load_lock()
    raw_root = local_root / "raw"
    checks: dict[str, bool] = {}
    errors: list[str] = []
    inventory: list[dict[str, Any]] = []
    for source in all_sources(lock, include_fallback=include_fallback):
        for domain, meta in source["files"].items():
            path = raw_root / source["submission"] / meta["name"]
            key = f"{source['agent']}:{domain}"
            exists = path.exists()
            checks[f"{key}.exists"] = exists
            if not exists:
                errors.append(f"missing {path}")
                continue
            digest = sha256_file(path)
            checks[f"{key}.sha256"] = digest == meta["sha256"] and path.stat().st_size == meta["bytes"]
            if not checks[f"{key}.sha256"]:
                errors.append(f"hash or size mismatch: {path}")
                continue
            try:
                result = read_json(path)
                file_errors = validate_result_file(result, source, domain, tau_root, lock)
            except Exception as exc:  # audit should report all bad files
                file_errors = [f"parse error: {exc}"]
            checks[f"{key}.content"] = not file_errors
            errors.extend(f"{key}: {error}" for error in file_errors)
            inventory.append({"agent": source["agent"], "submission": source["submission"], "domain": domain, "path": str(path), "sha256": digest, "bytes": path.stat().st_size, "errors": file_errors})
    audit = {"version": "v034-source-audit-1", "tau_version": lock["tau_version"], "passed": not errors and all(checks.values()), "checks": checks, "errors": errors, "inventory": inventory}
    write_json(local_root / "source_audit.json", audit)
    write_jsonl(local_root / "source_inventory.jsonl", inventory)
    return audit


def validate_result_file(result: dict[str, Any], source: dict[str, Any], domain: str, tau_root: Path | None, lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict) or not isinstance(result.get("simulations"), list) or not isinstance(result.get("tasks"), list):
        return ["missing tasks or simulations"]
    expected_tasks = lock["expected"][f"{domain}_tasks"]
    if len(result["tasks"]) != expected_tasks:
        errors.append(f"expected {expected_tasks} tasks, got {len(result['tasks'])}")
    simulations = result["simulations"]
    if len(simulations) != expected_tasks * lock["expected"]["trials"]:
        errors.append(f"expected {expected_tasks * lock['expected']['trials']} simulations, got {len(simulations)}")
    ids = [str(item.get("id")) for item in simulations]
    if len(ids) != len(set(ids)):
        errors.append("duplicate simulation ids")
    task_ids = {str(item.get("task_id")) for item in simulations}
    if len(task_ids) != expected_tasks:
        errors.append("simulation task coverage is incomplete")
    for simulation in simulations:
        errors.extend(validate_simulation(simulation))
    policy = ((result.get("info") or {}).get("environment_info") or {}).get("policy")
    if not isinstance(policy, str) or not policy.strip():
        errors.append("missing embedded policy")
    if tau_root is not None:
        current_task_path = tau_root / "data" / "tau2" / "domains" / domain / "tasks.json"
        current_policy_path = tau_root / "data" / "tau2" / "domains" / domain / "policy.md"
        if current_task_path.exists():
            current_tasks = read_json(current_task_path)
            embedded = {str(task["id"]): normalize_task(task) for task in result["tasks"]}
            current = {str(task["id"]): normalize_task(task) for task in current_tasks}
            if embedded != current:
                errors.append("embedded tasks differ from current τ-bench tasks")
        if current_policy_path.exists() and policy.strip() != current_policy_path.read_text(encoding="utf-8").strip():
            errors.append("embedded policy differs from current τ-bench policy")
    return errors


def validate_simulation(simulation: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = simulation.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["simulation has no messages"]
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"assistant", "user", "tool"}:
            errors.append("invalid message")
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
                errors.append("malformed tool call")
            else:
                if call["id"] in call_ids:
                    errors.append("duplicate tool call id")
                call_ids.add(call["id"])
        if message.get("role") == "tool":
            if not isinstance(message.get("id"), str):
                errors.append("tool result missing id")
            else:
                result_ids.add(message["id"])
    if not result_ids.issubset(call_ids):
        errors.append("tool result does not reference a call")
    reward_info = simulation.get("reward_info") or {}
    if not isinstance(reward_info.get("db_check"), dict):
        errors.append("missing db_check")
    return errors


def normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    drop = {"ticket", "issues", "required_documents", "user_tools", "annotations", "requestor", "compare_args", "env_assertions"}
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(child) for key, child in value.items() if key not in drop and child is not None}
        if isinstance(value, list):
            return [clean(child) for child in value]
        return value
    return clean(task)


def normalize_sources(local_root: Path = DEFAULT_LOCAL_ROOT, include_fallback: bool = False) -> dict[str, Any]:
    lock = load_lock()
    raw_root = local_root / "raw"
    trajectories: list[dict[str, Any]] = []
    screening: list[dict[str, Any]] = []
    policies: dict[str, dict[str, Any]] = {}
    tasks: dict[str, dict[str, Any]] = {}
    for source in all_sources(lock, include_fallback=include_fallback):
        for domain, meta in source["files"].items():
            path = raw_root / source["submission"] / meta["name"]
            result = read_json(path)
            env = (result.get("info") or {}).get("environment_info") or {}
            policy_text = str(env.get("policy", "")).strip()
            policy_id = content_id({"domain": domain, "policy": policy_text}, "pol_")
            policies[policy_id] = {"policy_id": policy_id, "domain": domain, "text": policy_text, "source_file": str(path)}
            for task in result["tasks"]:
                norm = normalize_task(task)
                task_id = content_id({"domain": domain, "task": norm}, "task_")
                tasks[task_id] = {"task_snapshot_id": task_id, "domain": domain, "task_id": str(task["id"]), "task": norm}
            for simulation in result["simulations"]:
                trajectory = normalize_simulation(simulation, source, domain, policy_id)
                trajectories.append(trajectory)
                reward_info = simulation.get("reward_info") or {}
                db_check = (reward_info.get("db_check") or {}).get("db_match")
                screening.append({"trajectory_id": trajectory["trajectory_id"], "domain": domain, "task_id": trajectory["task_id"], "source_agent": source["agent"], "trial": trajectory["trial"], "reward": reward_info.get("reward"), "db_match": db_check, "has_write": bool(trajectory["write_event_ids"]), "message_count": len(trajectory["events"]), "environment_errors": bool((reward_info.get("env_assertions") or [])), "reference_actions": reward_info.get("action_checks") or []})
    norm_root = local_root / "normalized"
    write_jsonl(norm_root / "trajectories.jsonl", trajectories)
    write_jsonl(norm_root / "screening_private.jsonl", screening)
    write_jsonl(norm_root / "policies.jsonl", list(policies.values()))
    write_jsonl(norm_root / "task_snapshots_private.jsonl", list(tasks.values()))
    manifest = {"version": "v034-normalized-1", "trajectory_count": len(trajectories), "policy_count": len(policies), "task_snapshot_count": len(tasks), "include_fallback": include_fallback}
    write_json(norm_root / "manifest.json", manifest)
    return manifest


def normalize_simulation(simulation: dict[str, Any], source: dict[str, Any], domain: str, policy_id: str) -> dict[str, Any]:
    base = {"source": source["submission"], "agent": source["agent"], "domain": domain, "task_id": str(simulation["task_id"]), "trial": simulation.get("trial"), "seed": simulation.get("seed"), "id": simulation.get("id")}
    trajectory_id = content_id(base, "traj_")
    events: list[dict[str, Any]] = []
    write_ids: list[str] = []
    result_for: dict[str, str] = {}
    for index, message in enumerate(simulation["messages"]):
        role = message.get("role")
        if message.get("content") is not None:
            event = {"event_id": content_id({"trajectory": trajectory_id, "index": index, "kind": "message", "content": message.get("content"), "role": role}, "evt_"), "turn": index, "role": role, "content": message.get("content"), "requestor": message.get("requestor")}
            events.append(event)
        for call_index, call in enumerate(message.get("tool_calls") or []):
            event_id = content_id({"trajectory": trajectory_id, "index": index, "call": call_index, "tool_call": call}, "evt_")
            event = {"event_id": event_id, "turn": index, "role": role, "tool_call_id": call.get("id"), "tool_call": {"name": call.get("name"), "arguments": call.get("arguments")}, "requestor": call.get("requestor")}
            events.append(event)
            if call.get("name") in WRITE_TOOLS.get(domain, set()) and call.get("requestor", "assistant") == "assistant":
                write_ids.append(event_id)
            result_for[str(call.get("id"))] = event_id
        if role == "tool":
            result_id = content_id({"trajectory": trajectory_id, "index": index, "kind": "tool_result", "id": message.get("id"), "content": message.get("content")}, "evt_")
            events.append({"event_id": result_id, "turn": index, "role": "tool", "tool_call_id": message.get("id"), "tool_result": {"content": message.get("content"), "error": bool(message.get("error", False))}, "requestor": message.get("requestor")})
    for event in events:
        if event.get("role") == "tool" and event.get("tool_call_id") in result_for:
            event["call_event_id"] = result_for[event["tool_call_id"]]
    return {"trajectory_id": trajectory_id, "source_file": source["submission"], "source_agent": source["agent"], "domain": domain, "task_id": str(simulation["task_id"]), "trial": simulation.get("trial"), "seed": simulation.get("seed"), "policy_id": policy_id, "events": events, "write_event_ids": write_ids}
