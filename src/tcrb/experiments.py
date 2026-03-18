from __future__ import annotations

import json
import statistics
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .benchmark import run_benchmark
from .config import benchmark_config_from_dict
from .models import BenchmarkConfig, PolicyMetrics, Workload
from .planner import PolicyNativePlanner, ToolPlanner


METRIC_FIELDS = [
    "task_success_rate",
    "invalid_tool_call_rate",
    "mean_latency_ms",
    "p95_latency_ms",
    "retries_per_successful_task",
    "estimated_cost_per_successful_task_usd",
]


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci95_half_width": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n": 0,
        }
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    ci95 = 1.96 * (std / (n**0.5)) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_half_width": ci95,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
        "n": n,
    }


def _policy_metric_map(rows: list[PolicyMetrics]) -> dict[str, PolicyMetrics]:
    return {row.policy: row for row in rows}


def parse_seed_list(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("seed list is empty")
    return [int(part) for part in parts]


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dict(existing, value)
            continue
        merged[key] = deepcopy(value)
    return merged


def run_multi_seed(
    workload: Workload,
    config: BenchmarkConfig,
    seeds: list[int],
    planner: ToolPlanner | None = None,
) -> dict:
    if not seeds:
        raise ValueError("seeds must not be empty")

    active_planner = planner or PolicyNativePlanner()
    per_seed: list[dict] = []
    aggregate: dict[str, dict[str, list[float]]] = {}

    for seed in seeds:
        seed_config = replace(config, seed=int(seed))
        result = run_benchmark(
            workload=workload, config=seed_config, planner=active_planner
        )
        policy_rows = _policy_metric_map(result.policy_metrics)

        per_seed.append(
            {
                "seed": int(seed),
                "policy_metrics": [asdict(metric) for metric in result.policy_metrics],
            }
        )

        for policy, row in policy_rows.items():
            entry = aggregate.setdefault(
                policy,
                {
                    "task_success_rate": [],
                    "invalid_tool_call_rate": [],
                    "mean_latency_ms": [],
                    "p95_latency_ms": [],
                    "retries_per_successful_task": [],
                    "estimated_cost_per_successful_task_usd": [],
                },
            )
            for field in METRIC_FIELDS:
                value = getattr(row, field)
                if value is None:
                    continue
                entry[field].append(float(value))

    aggregate_rows: list[dict] = []
    for policy, values in sorted(aggregate.items()):
        aggregate_rows.append(
            {
                "policy": policy,
                "metrics": {field: _stats(values[field]) for field in METRIC_FIELDS},
            }
        )

    return {
        "type": "multi_seed",
        "planner_id": getattr(active_planner, "planner_id", "planner"),
        "seeds": [int(seed) for seed in seeds],
        "per_seed": per_seed,
        "aggregate_policy_metrics": aggregate_rows,
    }


def run_sweep(
    workload: Workload,
    base_config_payload: dict,
    sweep_payload: dict,
    planner: ToolPlanner | None = None,
) -> dict:
    scenarios = list(sweep_payload.get("scenarios", []))
    if not scenarios:
        raise ValueError("sweep config must include at least one scenario")

    default_seeds = sweep_payload.get("seeds") or [int(base_config_payload["seed"])]
    scenario_results: list[dict] = []

    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        scenario_label = str(scenario.get("label", scenario_id))
        overrides = dict(scenario.get("config_overrides", {}))
        merged = deep_merge_dict(base_config_payload, overrides)
        scenario_config = benchmark_config_from_dict(merged)
        scenario_seeds = [int(seed) for seed in scenario.get("seeds", default_seeds)]

        multi_seed = run_multi_seed(
            workload=workload,
            config=scenario_config,
            seeds=scenario_seeds,
            planner=planner,
        )
        scenario_results.append(
            {
                "id": scenario_id,
                "label": scenario_label,
                "config_overrides": overrides,
                "result": multi_seed,
            }
        )

    return {
        "type": "sweep",
        "name": str(sweep_payload.get("name", "sweep")),
        "description": str(sweep_payload.get("description", "")).strip(),
        "planner_id": getattr(
            planner or PolicyNativePlanner(), "planner_id", "planner"
        ),
        "scenarios": scenario_results,
    }


def write_json(payload: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
