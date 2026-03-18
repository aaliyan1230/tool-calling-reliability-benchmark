from dataclasses import replace

from tcrb.benchmark import run_benchmark
from tcrb.models import BenchmarkConfig, CostModel, TaskSpec, ToolSpec, Workload


def _base_config(**overrides):
    config = BenchmarkConfig(
        seed=7,
        max_attempts=3,
        default_timeout_ms=100,
        time_budget_ms=500,
        base_backoff_ms=20,
        backoff_jitter_ms=0,
        fault_probabilities={
            "timeout": 0.0,
            "rate_limit": 0.0,
            "malformed_schema": 0.0,
            "contract_drift": 0.0,
            "network_failure": 0.0,
        },
        policies=[
            "naive_retry",
            "exponential_backoff_jitter",
            "schema_first_fallback",
            "timeout_budget_early_abort",
        ],
        cost=CostModel(base_per_call_usd=0.001, per_ms_usd=0.0),
    )
    return replace(config, **overrides)


def test_benchmark_returns_metrics_for_each_policy():
    workload = Workload(
        tools={
            "tool_a": ToolSpec(
                "tool_a", base_latency_ms=10, jitter_ms=0, schema_fields=["answer"]
            ),
            "tool_b": ToolSpec(
                "tool_b",
                base_latency_ms=12,
                jitter_ms=0,
                schema_fields=["answer", "debug"],
            ),
        },
        tasks=[
            TaskSpec(
                task_id="t1",
                primary_tool="tool_a",
                fallback_tools=["tool_b"],
                required_schema=["answer"],
            ),
            TaskSpec(
                task_id="t2",
                primary_tool="tool_a",
                fallback_tools=["tool_b"],
                required_schema=["answer", "debug"],
            ),
        ],
    )

    result = run_benchmark(workload, _base_config())

    assert len(result.policy_metrics) == 4
    assert len(result.task_results) == 8
    assert all(metric.tasks_total == 2 for metric in result.policy_metrics)


def test_schema_first_uses_fallback_after_schema_fault():
    task = TaskSpec(
        task_id="schema-fallback",
        primary_tool="primary",
        fallback_tools=["fallback"],
        required_schema=["answer"],
    )
    workload = Workload(
        tools={
            "primary": ToolSpec(
                name="primary",
                base_latency_ms=20,
                jitter_ms=0,
                schema_fields=["answer"],
                fault_multipliers={"malformed_schema": 1.0},
            ),
            "fallback": ToolSpec(
                name="fallback",
                base_latency_ms=20,
                jitter_ms=0,
                schema_fields=["answer"],
                fault_multipliers={"malformed_schema": 0.0},
            ),
        },
        tasks=[task],
    )
    config = _base_config(
        policies=["schema_first_fallback"],
        fault_probabilities={
            "timeout": 0.0,
            "rate_limit": 0.0,
            "malformed_schema": 1.0,
            "contract_drift": 0.0,
            "network_failure": 0.0,
        },
    )

    result = run_benchmark(workload, config)
    task_result = result.task_results[0]

    assert task_result.success is True
    assert len(task_result.attempts) == 2
    assert task_result.attempts[0].tool_name == "primary"
    assert task_result.attempts[0].status == "malformed_schema"
    assert task_result.attempts[1].tool_name == "fallback"
    assert task_result.attempts[1].status == "success"


def test_timeout_budget_policy_aborts_when_budget_exhausted():
    task = TaskSpec(
        task_id="budget", primary_tool="slow", fallback_tools=[], required_schema=["ok"]
    )
    workload = Workload(
        tools={
            "slow": ToolSpec(
                name="slow",
                base_latency_ms=100,
                jitter_ms=0,
                schema_fields=["ok"],
                timeout_ms=40,
            ),
        },
        tasks=[task],
    )
    config = _base_config(
        policies=["timeout_budget_early_abort"],
        time_budget_ms=45,
        fault_probabilities={
            "timeout": 1.0,
            "rate_limit": 0.0,
            "malformed_schema": 0.0,
            "contract_drift": 0.0,
            "network_failure": 0.0,
        },
    )

    result = run_benchmark(workload, config)
    task_result = result.task_results[0]

    assert task_result.success is False
    assert task_result.final_status == "budget_exhausted"
    assert len(task_result.attempts) == 1
