import pytest

from tcrb.finetune.dataset import build_examples_from_result_payload, split_examples
from tcrb.finetune.evaluate import compare_run_payloads


def test_build_examples_from_result_payload_success_only():
    payload = {
        "task_results": [
            {
                "task_id": "t1",
                "policy": "schema_first_fallback",
                "planner_id": "command",
                "attempts": [
                    {
                        "attempt_number": 1,
                        "tool_name": "a",
                        "status": "timeout",
                        "invalid_tool_call": False,
                    },
                    {
                        "attempt_number": 2,
                        "tool_name": "b",
                        "status": "success",
                        "invalid_tool_call": False,
                    },
                ],
            }
        ]
    }

    rows = build_examples_from_result_payload(payload)

    assert len(rows) == 1
    assert rows[0]["completion"]["tool_name"] == "b"
    assert rows[0]["prompt"]["attempted_tools"] == ["a"]


def test_split_examples_respects_validation_ratio():
    rows = [{"id": idx} for idx in range(10)]
    train, eval_rows = split_examples(rows, 0.2, seed=1)

    assert len(train) == 8
    assert len(eval_rows) == 2


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
    finetuned = {
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

    result = compare_run_payloads(base, finetuned)

    assert result["policies"][0]["policy"] == "naive_retry"
    assert result["policies"][0]["delta"]["task_success_rate"] == pytest.approx(0.2)
