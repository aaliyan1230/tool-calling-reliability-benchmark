from __future__ import annotations

import json
from pathlib import Path

from .models import BenchmarkConfig, CostModel, TaskSpec, ToolSpec, Workload


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_workload(path: str | Path) -> Workload:
    payload = _read_json(path)
    tools: dict[str, ToolSpec] = {}
    for tool in payload["tools"]:
        spec = ToolSpec(
            name=tool["name"],
            base_latency_ms=int(tool["base_latency_ms"]),
            jitter_ms=int(tool.get("jitter_ms", 0)),
            schema_fields=list(tool["schema_fields"]),
            timeout_ms=tool.get("timeout_ms"),
            fault_multipliers=dict(tool.get("fault_multipliers", {})),
        )
        tools[spec.name] = spec

    tasks = [
        TaskSpec(
            task_id=item["task_id"],
            primary_tool=item["primary_tool"],
            fallback_tools=list(item.get("fallback_tools", [])),
            required_schema=list(item["required_schema"]),
        )
        for item in payload["tasks"]
    ]
    return Workload(tools=tools, tasks=tasks)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    payload = _read_json(path)
    return BenchmarkConfig(
        seed=int(payload["seed"]),
        max_attempts=int(payload["max_attempts"]),
        default_timeout_ms=int(payload["default_timeout_ms"]),
        time_budget_ms=int(payload["time_budget_ms"]),
        base_backoff_ms=int(payload["base_backoff_ms"]),
        backoff_jitter_ms=int(payload["backoff_jitter_ms"]),
        fault_probabilities=dict(payload["fault_probabilities"]),
        policies=list(payload["policies"]),
        cost=CostModel(
            base_per_call_usd=float(payload["cost"]["base_per_call_usd"]),
            per_ms_usd=float(payload["cost"]["per_ms_usd"]),
        ),
        retryable_faults=list(
            payload.get(
                "retryable_faults", ["timeout", "rate_limit", "network_failure"]
            )
        ),
        policy_overrides=dict(payload.get("policy_overrides", {})),
    )
