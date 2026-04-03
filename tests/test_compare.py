import pytest

from tcrb.compare import compare_run_payloads


def test_compare_run_payloads_handles_multi_seed_shape():
    base = {
        "type": "multi_seed",
        "aggregate_policy_metrics": [
            {
                "policy": "naive_retry",
                "metrics": {
                    "task_success_rate": {"mean": 0.5},
                    "invalid_tool_call_rate": {"mean": 0.1},
                    "mean_latency_ms": {"mean": 100.0},
                    "p95_latency_ms": {"mean": 120.0},
                    "retries_per_successful_task": {"mean": 1.0},
                    "estimated_cost_per_successful_task_usd": {"mean": 0.02},
                },
            }
        ],
    }
    comparison = {
        "type": "multi_seed",
        "aggregate_policy_metrics": [
            {
                "policy": "naive_retry",
                "metrics": {
                    "task_success_rate": {"mean": 0.7},
                    "invalid_tool_call_rate": {"mean": 0.08},
                    "mean_latency_ms": {"mean": 90.0},
                    "p95_latency_ms": {"mean": 110.0},
                    "retries_per_successful_task": {"mean": 0.8},
                    "estimated_cost_per_successful_task_usd": {"mean": 0.015},
                },
            }
        ],
    }

    result = compare_run_payloads(base, comparison)

    assert result["policies"][0]["policy"] == "naive_retry"
    assert result["policies"][0]["delta"]["task_success_rate"] == pytest.approx(0.2)
    assert result["policies"][0]["comparison"]["task_success_rate"] == pytest.approx(0.7)
