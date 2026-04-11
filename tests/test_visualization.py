from pathlib import Path

from tcrb.visualization import AnalysisPayloads, render_analysis_plots


def _multi_seed_payload() -> dict:
    return {
        "type": "multi_seed",
        "aggregate_policy_metrics": [
            {
                "policy": "naive_retry",
                "metrics": {
                    "task_success_rate": {"mean": 0.61, "ci95_half_width": 0.02},
                    "invalid_tool_call_rate": {"mean": 0.29, "ci95_half_width": 0.03},
                    "p95_latency_ms": {"mean": 1336.0, "ci95_half_width": 20.0},
                    "estimated_cost_per_successful_task_usd": {
                        "mean": 0.00159,
                        "ci95_half_width": 0.0001,
                    },
                },
            },
            {
                "policy": "schema_first_fallback",
                "metrics": {
                    "task_success_rate": {"mean": 0.88, "ci95_half_width": 0.01},
                    "invalid_tool_call_rate": {"mean": 0.16, "ci95_half_width": 0.02},
                    "p95_latency_ms": {"mean": 976.0, "ci95_half_width": 15.0},
                    "estimated_cost_per_successful_task_usd": {
                        "mean": 0.00123,
                        "ci95_half_width": 0.0001,
                    },
                },
            },
        ],
    }


def _delta_payload() -> dict:
    return {
        "target": {
            "policies": [
                {
                    "policy": "naive_retry",
                    "delta": {
                        "task_success_rate": 0.27,
                        "invalid_tool_call_rate": -0.20,
                    },
                },
                {
                    "policy": "timeout_budget_early_abort",
                    "delta": {
                        "task_success_rate": 0.22,
                        "invalid_tool_call_rate": -0.28,
                    },
                },
            ]
        }
    }


def _matrix_payload() -> dict:
    return {
        "portfolio_verdict": "FAIL",
        "rows": [
            {
                "toolset_id": "customer_support",
                "split": "target",
                "verdict": "FAIL",
                "delta_first_tool_accuracy": -0.42,
                "delta_sequence_prefix_accuracy": -0.42,
            },
            {
                "toolset_id": "ecommerce_ops",
                "split": "open",
                "verdict": "FAIL",
                "delta_first_tool_accuracy": -0.56,
                "delta_sequence_prefix_accuracy": -0.57,
            },
        ],
    }


def test_render_analysis_plots_writes_expected_pngs(tmp_path: Path):
    outputs = render_analysis_plots(
        AnalysisPayloads(
            multi_seed=_multi_seed_payload(),
            delta=_delta_payload(),
            matrix=_matrix_payload(),
        ),
        tmp_path,
    )

    assert set(outputs.keys()) == {"multi_seed", "delta", "matrix"}
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0