from tcrb.reporting import (
    render_delta_markdown,
    render_multi_seed_markdown,
    render_study_gate_markdown,
)
from tcrb.transfer_matrix import MatrixThresholds, render_transfer_matrix_markdown


def test_render_delta_markdown_includes_snapshot_and_assets():
    payload = {
        "target": {
            "policies": [
                {
                    "policy": "naive_retry",
                    "delta": {
                        "task_success_rate": 0.2,
                        "invalid_tool_call_rate": -0.1,
                        "mean_latency_ms": 0.0,
                        "p95_latency_ms": 0.0,
                        "retries_per_successful_task": 0.0,
                        "estimated_cost_per_successful_task_usd": 0.0,
                    },
                }
            ]
        }
    }

    text = render_delta_markdown(payload, asset_paths={"delta_plot": "delta_policy.png"})

    assert "### Verdict Snapshot" in text
    assert "mean_success_delta: +0.2000" in text
    assert "![delta_plot](delta_policy.png)" in text


def test_render_multi_seed_markdown_includes_asset_block():
    payload = {
        "seeds": [1, 2, 3],
        "aggregate_policy_metrics": [
            {
                "policy": "heuristic",
                "metrics": {
                    "task_success_rate": {"mean": 0.8, "ci95_half_width": 0.05},
                    "invalid_tool_call_rate": {"mean": 0.1, "ci95_half_width": 0.01},
                    "mean_latency_ms": {"mean": 100.0, "ci95_half_width": 5.0},
                    "p95_latency_ms": {"mean": 140.0, "ci95_half_width": 7.0},
                    "retries_per_successful_task": {"mean": 0.2, "ci95_half_width": 0.02},
                    "estimated_cost_per_successful_task_usd": {"mean": 0.001, "ci95_half_width": 0.0001},
                },
            }
        ],
    }

    text = render_multi_seed_markdown(payload, asset_paths={"overview": "multi_seed_overview.png"})

    assert "best_success_policy: heuristic (0.8000)" in text
    assert "![overview](multi_seed_overview.png)" in text


def test_render_study_gate_markdown_includes_assets():
    payload = {
        "verdict": "PASS",
        "checks": [{"name": "flatline", "passed": True, "value": 0.1, "threshold": 0.001, "detail": "ok"}],
    }

    text = render_study_gate_markdown(
        payload,
        asset_paths={"delta_plot": "delta_policy.png", "matrix_plot": "transfer_matrix.png"},
    )

    assert "Verdict: PASS" in text
    assert "![delta_plot](delta_policy.png)" in text
    assert "![matrix_plot](transfer_matrix.png)" in text


def test_render_transfer_matrix_markdown_includes_snapshot_and_asset():
    text = render_transfer_matrix_markdown(
        target_toolset_id="customer_support",
        rows=[
            {
                "toolset_id": "customer_support",
                "split": "target",
                "base_first_tool_accuracy": 1.0,
                "comparison_first_tool_accuracy": 0.5,
                "delta_first_tool_accuracy": -0.5,
                "base_sequence_prefix_accuracy": 1.0,
                "comparison_sequence_prefix_accuracy": 0.5,
                "delta_sequence_prefix_accuracy": -0.5,
                "verdict": "FAIL",
            }
        ],
        thresholds=MatrixThresholds(),
        asset_paths={"matrix_plot": "transfer_matrix.png"},
    )

    assert "portfolio_verdict: FAIL" in text
    assert "worst_toolset: customer_support" in text
    assert "![Transfer matrix asset](transfer_matrix.png)" in text