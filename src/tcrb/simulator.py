from __future__ import annotations

import random
from dataclasses import dataclass

from .models import BenchmarkConfig, FAULT_ORDER, TaskSpec, ToolSpec


SCHEMA_FAULTS = {"malformed_schema", "contract_drift", "invalid_schema"}


@dataclass(frozen=True)
class CallOutcome:
    status: str
    latency_ms: int
    schema_valid: bool
    invalid_tool_call: bool


def _sample_fault(
    tool: ToolSpec, config: BenchmarkConfig, rng: random.Random
) -> str | None:
    weighted_faults: list[tuple[str, float]] = []
    total_probability = 0.0
    for name in FAULT_ORDER:
        base = float(config.fault_probabilities.get(name, 0.0))
        multiplier = float(tool.fault_multipliers.get(name, 1.0))
        value = max(0.0, base * multiplier)
        if value > 0:
            weighted_faults.append((name, value))
            total_probability += value

    if not weighted_faults:
        return None

    if total_probability >= 1.0:
        pick = rng.random() * total_probability
    else:
        pick = rng.random()
        if pick > total_probability:
            return None

    cursor = 0.0
    for fault_name, weight in weighted_faults:
        cursor += weight
        if pick <= cursor:
            return fault_name
    return weighted_faults[-1][0]


def _base_latency_ms(tool: ToolSpec, rng: random.Random) -> int:
    jitter = rng.randint(-tool.jitter_ms, tool.jitter_ms) if tool.jitter_ms > 0 else 0
    return max(1, tool.base_latency_ms + jitter)


def simulate_call(
    tool: ToolSpec,
    task: TaskSpec,
    config: BenchmarkConfig,
    rng: random.Random,
) -> CallOutcome:
    fault = _sample_fault(tool, config, rng)
    raw_latency = _base_latency_ms(tool, rng)
    timeout_limit = int(tool.timeout_ms or config.default_timeout_ms)

    if fault == "timeout":
        return CallOutcome(
            status="timeout",
            latency_ms=max(raw_latency, timeout_limit + rng.randint(1, 120)),
            schema_valid=False,
            invalid_tool_call=False,
        )
    if fault == "rate_limit":
        return CallOutcome(
            status="rate_limit",
            latency_ms=max(5, raw_latency // 3),
            schema_valid=False,
            invalid_tool_call=False,
        )
    if fault == "network_failure":
        return CallOutcome(
            status="network_failure",
            latency_ms=max(8, raw_latency // 2),
            schema_valid=False,
            invalid_tool_call=False,
        )

    if fault == "malformed_schema":
        return CallOutcome(
            status="malformed_schema",
            latency_ms=raw_latency,
            schema_valid=False,
            invalid_tool_call=True,
        )

    if fault == "contract_drift":
        return CallOutcome(
            status="contract_drift",
            latency_ms=raw_latency,
            schema_valid=False,
            invalid_tool_call=True,
        )

    if raw_latency > timeout_limit:
        return CallOutcome(
            status="timeout",
            latency_ms=timeout_limit + rng.randint(1, 60),
            schema_valid=False,
            invalid_tool_call=False,
        )

    schema_valid = all(field in tool.schema_fields for field in task.required_schema)
    if not schema_valid:
        return CallOutcome(
            status="invalid_schema",
            latency_ms=raw_latency,
            schema_valid=False,
            invalid_tool_call=True,
        )

    return CallOutcome(
        status="success",
        latency_ms=raw_latency,
        schema_valid=True,
        invalid_tool_call=False,
    )
