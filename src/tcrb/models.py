from __future__ import annotations

from dataclasses import dataclass, field


FAULT_ORDER = [
    "timeout",
    "rate_limit",
    "malformed_schema",
    "contract_drift",
    "network_failure",
]


@dataclass(frozen=True)
class CostModel:
    base_per_call_usd: float
    per_ms_usd: float


@dataclass(frozen=True)
class ToolSpec:
    name: str
    base_latency_ms: int
    jitter_ms: int
    schema_fields: list[str]
    timeout_ms: int | None = None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    primary_tool: str
    fallback_tools: list[str]
    required_schema: list[str]


@dataclass(frozen=True)
class Workload:
    tools: dict[str, ToolSpec]
    tasks: list[TaskSpec]


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int
    max_attempts: int
    default_timeout_ms: int
    time_budget_ms: int
    base_backoff_ms: int
    backoff_jitter_ms: int
    fault_probabilities: dict[str, float]
    policies: list[str]
    cost: CostModel
    retryable_faults: list[str] = field(
        default_factory=lambda: ["timeout", "rate_limit", "network_failure"]
    )
    policy_overrides: dict[str, dict] = field(default_factory=dict)


@dataclass
class AttemptRecord:
    task_id: str
    policy: str
    attempt_number: int
    tool_name: str
    status: str
    schema_valid: bool
    invalid_tool_call: bool
    latency_ms: int
    retry_delay_ms: int
    cost_usd: float


@dataclass
class TaskResult:
    task_id: str
    policy: str
    success: bool
    final_status: str
    total_latency_ms: int
    total_cost_usd: float
    retries: int
    attempts: list[AttemptRecord]


@dataclass
class PolicyMetrics:
    policy: str
    tasks_total: int
    tasks_succeeded: int
    task_success_rate: float
    invalid_tool_call_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    retries_per_successful_task: float
    estimated_cost_per_successful_task_usd: float | None


@dataclass
class BenchmarkResult:
    policy_metrics: list[PolicyMetrics]
    task_results: list[TaskResult]
