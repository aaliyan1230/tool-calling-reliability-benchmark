from tcrb.config import benchmark_config_from_dict, workload_from_dict
from tcrb.experiments import deep_merge_dict, run_multi_seed, run_sweep


def _base_config_payload() -> dict:
    return {
        "seed": 5,
        "max_attempts": 3,
        "default_timeout_ms": 100,
        "time_budget_ms": 400,
        "base_backoff_ms": 20,
        "backoff_jitter_ms": 5,
        "fault_probabilities": {
            "timeout": 0.05,
            "rate_limit": 0.03,
            "malformed_schema": 0.02,
            "contract_drift": 0.01,
            "network_failure": 0.02,
        },
        "retryable_faults": ["timeout", "rate_limit", "network_failure"],
        "policies": ["naive_retry", "schema_first_fallback"],
        "policy_overrides": {},
        "cost": {"base_per_call_usd": 0.001, "per_ms_usd": 0.0},
    }


def _workload_payload() -> dict:
    return {
        "tools": [
            {
                "name": "alpha",
                "base_latency_ms": 10,
                "jitter_ms": 0,
                "schema_fields": ["answer"],
                "fault_multipliers": {},
            },
            {
                "name": "beta",
                "base_latency_ms": 12,
                "jitter_ms": 0,
                "schema_fields": ["answer", "debug"],
                "fault_multipliers": {},
            },
        ],
        "tasks": [
            {
                "task_id": "t1",
                "primary_tool": "alpha",
                "fallback_tools": ["beta"],
                "required_schema": ["answer"],
            }
        ],
    }


def test_deep_merge_preserves_nested_values():
    base = {
        "fault_probabilities": {"timeout": 0.1, "rate_limit": 0.1},
        "max_attempts": 4,
    }
    override = {"fault_probabilities": {"timeout": 0.2}}

    merged = deep_merge_dict(base, override)

    assert merged["fault_probabilities"]["timeout"] == 0.2
    assert merged["fault_probabilities"]["rate_limit"] == 0.1
    assert merged["max_attempts"] == 4


def test_run_multi_seed_outputs_aggregate_rows():
    workload = workload_from_dict(_workload_payload())
    config = benchmark_config_from_dict(_base_config_payload())

    payload = run_multi_seed(workload=workload, config=config, seeds=[1, 2, 3])

    assert payload["type"] == "multi_seed"
    assert payload["seeds"] == [1, 2, 3]
    assert len(payload["per_seed"]) == 3
    policies = {row["policy"] for row in payload["aggregate_policy_metrics"]}
    assert policies == {"naive_retry", "schema_first_fallback"}


def test_run_sweep_applies_overrides_by_scenario():
    workload = workload_from_dict(_workload_payload())
    base_payload = _base_config_payload()
    sweep_payload = {
        "name": "unit-sweep",
        "seeds": [9, 10],
        "scenarios": [
            {
                "id": "s1",
                "config_overrides": {"fault_probabilities": {"timeout": 0.0}},
            },
            {
                "id": "s2",
                "config_overrides": {"fault_probabilities": {"timeout": 0.3}},
            },
        ],
    }

    payload = run_sweep(
        workload=workload, base_config_payload=base_payload, sweep_payload=sweep_payload
    )

    assert payload["type"] == "sweep"
    assert payload["name"] == "unit-sweep"
    assert len(payload["scenarios"]) == 2
    assert payload["scenarios"][0]["id"] == "s1"
    assert payload["scenarios"][1]["id"] == "s2"
    assert payload["scenarios"][0]["result"]["seeds"] == [9, 10]
