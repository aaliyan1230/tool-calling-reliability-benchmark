import json
import sys
from pathlib import Path

from tcrb.cli import main


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def _study_gate_payload() -> dict:
    return {
        "verdict": "PASS",
        "checks": [
            {
                "name": "base_vs_comparison_nonflatline",
                "passed": True,
                "value": 0.28,
                "threshold": 0.0001,
            }
        ],
    }


def _write_standard_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "multi_seed.json", _multi_seed_payload())
    _write_json(run_dir / "delta-ms.json", _delta_payload())
    _write_json(run_dir / "matrix.json", _matrix_payload())
    _write_json(run_dir / "study_gate.json", _study_gate_payload())


def test_render_plots_cli_renders_from_run_dir(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "artifacts"
    _write_standard_artifacts(run_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        ["tcrb", "render-plots", "--run-dir", str(run_dir)],
    )

    assert main() == 0
    assert (run_dir / "multi_seed_overview.png").exists()
    assert (run_dir / "delta_policy.png").exists()
    assert (run_dir / "transfer_matrix.png").exists()


def test_summarize_run_cli_writes_markdown_with_plot_refs(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "artifacts"
    _write_standard_artifacts(run_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tcrb",
            "summarize-run",
            "--run-dir",
            str(run_dir),
            "--output-report",
            str(run_dir / "analysis_summary.md"),
        ],
    )

    assert main() == 0

    summary = (run_dir / "analysis_summary.md").read_text(encoding="utf-8")
    assert "study_gate_verdict: PASS" in summary
    assert "matrix_portfolio_verdict: FAIL" in summary
    assert "![Delta policy view](delta_policy.png)" in summary
    assert "![Transfer matrix view](transfer_matrix.png)" in summary