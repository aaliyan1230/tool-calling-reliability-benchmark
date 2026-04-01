from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METRIC_NAMES = [
    "task_success_rate",
    "invalid_tool_call_rate",
    "mean_latency_ms",
    "p95_latency_ms",
    "retries_per_successful_task",
    "estimated_cost_per_successful_task_usd",
]


def load_json_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_policy_metrics(payload: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    if payload.get("type") == "multi_seed":
        rows = payload.get("aggregate_policy_metrics", [])
        normalized: dict[str, dict[str, float | None]] = {}
        for row in rows:
            policy = str(row.get("policy", ""))
            metrics = row.get("metrics", {})
            normalized[policy] = {
                metric: (
                    float(metrics.get(metric, {}).get("mean"))
                    if metrics.get(metric, {}).get("mean") is not None
                    else None
                )
                for metric in METRIC_NAMES
            }
        return normalized

    rows = payload.get("policy_metrics", [])
    normalized = {}
    for row in rows:
        policy = str(row.get("policy", ""))
        normalized[policy] = {
            metric: (
                float(row.get(metric)) if row.get(metric) is not None else None
            )
            for metric in METRIC_NAMES
        }
    return normalized


def compare_run_payloads(
    base_payload: dict[str, Any],
    finetuned_payload: dict[str, Any],
) -> dict[str, Any]:
    base = _normalize_policy_metrics(base_payload)
    finetuned = _normalize_policy_metrics(finetuned_payload)

    policies = sorted(set(base.keys()) | set(finetuned.keys()))
    rows: list[dict[str, Any]] = []
    for policy in policies:
        base_metrics = base.get(policy, {})
        ft_metrics = finetuned.get(policy, {})

        delta = {}
        for metric in METRIC_NAMES:
            b = base_metrics.get(metric)
            f = ft_metrics.get(metric)
            delta[metric] = None if (b is None or f is None) else float(f - b)

        rows.append(
            {
                "policy": policy,
                "base": base_metrics,
                "finetuned": ft_metrics,
                "delta": delta,
            }
        )

    return {
        "policies": rows,
    }
