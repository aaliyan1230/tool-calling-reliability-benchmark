from __future__ import annotations

import random
from dataclasses import replace

from .models import (
    AttemptRecord,
    BenchmarkConfig,
    TaskResult,
    TaskSpec,
    ToolSpec,
    Workload,
)
from .simulator import SCHEMA_FAULTS, simulate_call


def _candidate_tool_names(task: TaskSpec, workload: Workload) -> list[str]:
    ordered = [task.primary_tool, *task.fallback_tools]
    return [name for name in ordered if name in workload.tools]


def _supports_schema(tool: ToolSpec, task: TaskSpec) -> bool:
    return all(field in tool.schema_fields for field in task.required_schema)


def _select_initial_tool(policy: str, task: TaskSpec, workload: Workload) -> str:
    candidates = _candidate_tool_names(task, workload)
    if not candidates:
        raise ValueError(f"Task {task.task_id} has no valid tools")

    if policy == "schema_first_fallback":
        for name in candidates:
            if _supports_schema(workload.tools[name], task):
                return name
    return candidates[0]


def _next_schema_compatible_tool(
    task: TaskSpec, workload: Workload, attempted: set[str]
) -> str | None:
    for name in _candidate_tool_names(task, workload):
        if name in attempted:
            continue
        if _supports_schema(workload.tools[name], task):
            return name
    return None


def _retry_delay_ms(
    policy: str, attempt_number: int, config: BenchmarkConfig, rng: random.Random
) -> int:
    if policy == "exponential_backoff_jitter":
        base = config.base_backoff_ms * (2 ** max(0, attempt_number - 1))
        jitter = rng.randint(0, max(0, config.backoff_jitter_ms))
        return max(0, int(base + jitter))
    if policy == "timeout_budget_early_abort":
        jitter = rng.randint(0, max(0, config.backoff_jitter_ms // 2))
        return max(0, int(config.base_backoff_ms + jitter))
    return 0


def _can_retry(status: str, config: BenchmarkConfig) -> bool:
    return status in config.retryable_faults


def apply_policy_override(config: BenchmarkConfig, policy: str) -> BenchmarkConfig:
    override = dict(config.policy_overrides.get(policy, {}))
    if not override:
        return config

    mutable_fields = {
        "seed",
        "max_attempts",
        "default_timeout_ms",
        "time_budget_ms",
        "base_backoff_ms",
        "backoff_jitter_ms",
        "fault_probabilities",
        "policies",
        "retryable_faults",
        "policy_overrides",
    }
    kwargs = {k: v for k, v in override.items() if k in mutable_fields}
    return replace(config, **kwargs)


def run_task_for_policy(
    policy: str,
    task: TaskSpec,
    workload: Workload,
    config: BenchmarkConfig,
    rng: random.Random,
) -> TaskResult:
    effective = apply_policy_override(config, policy)
    attempts: list[AttemptRecord] = []
    elapsed_ms = 0
    total_cost = 0.0
    attempted_tools: set[str] = set()
    tool_name = _select_initial_tool(policy, task, workload)

    success = False
    final_status = "unknown"

    for attempt_number in range(1, effective.max_attempts + 1):
        tool = workload.tools[tool_name]
        outcome = simulate_call(tool=tool, task=task, config=effective, rng=rng)

        elapsed_ms += outcome.latency_ms
        cost = effective.cost.base_per_call_usd + (
            effective.cost.per_ms_usd * outcome.latency_ms
        )
        total_cost += cost
        attempted_tools.add(tool_name)

        record = AttemptRecord(
            task_id=task.task_id,
            policy=policy,
            attempt_number=attempt_number,
            tool_name=tool_name,
            status=outcome.status,
            schema_valid=outcome.schema_valid,
            invalid_tool_call=outcome.invalid_tool_call,
            latency_ms=outcome.latency_ms,
            retry_delay_ms=0,
            cost_usd=cost,
        )
        attempts.append(record)

        if outcome.status == "success":
            success = True
            final_status = "success"
            break

        final_status = outcome.status
        if attempt_number >= effective.max_attempts:
            break

        should_retry = _can_retry(outcome.status, effective)

        if policy == "schema_first_fallback" and outcome.status in SCHEMA_FAULTS:
            replacement_tool = _next_schema_compatible_tool(
                task, workload, attempted_tools
            )
            if replacement_tool:
                tool_name = replacement_tool
                should_retry = True
            else:
                should_retry = False
        elif outcome.status in SCHEMA_FAULTS:
            should_retry = False

        if policy == "timeout_budget_early_abort":
            remaining = effective.time_budget_ms - elapsed_ms
            if remaining <= 0:
                final_status = "budget_exhausted"
                break

        if not should_retry:
            break

        retry_delay = _retry_delay_ms(policy, attempt_number, effective, rng)
        if policy == "timeout_budget_early_abort":
            remaining = effective.time_budget_ms - elapsed_ms
            if retry_delay >= remaining:
                final_status = "early_abort"
                break

        attempts[-1].retry_delay_ms = retry_delay
        elapsed_ms += retry_delay

    return TaskResult(
        task_id=task.task_id,
        policy=policy,
        success=success,
        final_status=final_status,
        total_latency_ms=elapsed_ms,
        total_cost_usd=total_cost,
        retries=max(0, len(attempts) - 1),
        attempts=attempts,
    )
