from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .models import (
    BenchmarkConfig,
    BenchmarkResult,
    PolicyMetrics,
    TaskResult,
    Workload,
)
from .policies import run_task_for_policy


def _p95(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round(0.95 * (len(ordered) - 1)))
    return float(ordered[index])


def _policy_metrics(policy: str, task_results: list[TaskResult]) -> PolicyMetrics:
    tasks_total = len(task_results)
    successes = [r for r in task_results if r.success]
    tasks_succeeded = len(successes)
    total_attempts = sum(len(r.attempts) for r in task_results)
    invalid_attempts = sum(
        1 for r in task_results for a in r.attempts if a.invalid_tool_call
    )

    latencies = [r.total_latency_ms for r in task_results]
    success_rate = tasks_succeeded / tasks_total if tasks_total else 0.0
    invalid_rate = invalid_attempts / total_attempts if total_attempts else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = _p95(latencies)
    retries_total = sum(r.retries for r in task_results)
    retries_per_success = retries_total / tasks_succeeded if tasks_succeeded else 0.0

    if tasks_succeeded:
        success_cost = sum(r.total_cost_usd for r in successes)
        cost_per_success = success_cost / tasks_succeeded
    else:
        cost_per_success = None

    return PolicyMetrics(
        policy=policy,
        tasks_total=tasks_total,
        tasks_succeeded=tasks_succeeded,
        task_success_rate=success_rate,
        invalid_tool_call_rate=invalid_rate,
        mean_latency_ms=mean_latency,
        p95_latency_ms=p95_latency,
        retries_per_successful_task=retries_per_success,
        estimated_cost_per_successful_task_usd=cost_per_success,
    )


def run_benchmark(workload: Workload, config: BenchmarkConfig) -> BenchmarkResult:
    rng = random.Random(config.seed)
    all_task_results: list[TaskResult] = []
    all_policy_metrics: list[PolicyMetrics] = []

    for policy in config.policies:
        policy_results: list[TaskResult] = []
        for task in workload.tasks:
            result = run_task_for_policy(
                policy=policy,
                task=task,
                workload=workload,
                config=config,
                rng=rng,
            )
            policy_results.append(result)
            all_task_results.append(result)

        all_policy_metrics.append(_policy_metrics(policy, policy_results))

    return BenchmarkResult(
        policy_metrics=all_policy_metrics, task_results=all_task_results
    )


def benchmark_to_dict(result: BenchmarkResult) -> dict:
    return {
        "policy_metrics": [asdict(m) for m in result.policy_metrics],
        "task_results": [asdict(r) for r in result.task_results],
    }


def write_result_json(result: BenchmarkResult, output_path: str | Path) -> None:
    payload = benchmark_to_dict(result)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
