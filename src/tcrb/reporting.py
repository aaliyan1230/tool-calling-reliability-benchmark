from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .models import BenchmarkResult, PolicyMetrics


def _fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _append_asset_section(lines: list[str], asset_paths: dict[str, str] | None) -> None:
    if not asset_paths:
        return

    lines.extend(["", "### Assets", ""])
    for label, path in asset_paths.items():
        lines.append(f"- {label}: `{path}`")


def _append_asset_images(lines: list[str], asset_paths: dict[str, str] | None) -> None:
    if not asset_paths:
        return

    for label, path in asset_paths.items():
        if str(path).lower().endswith(".png"):
            lines.extend(["", f"![{label}]({path})"])


def policy_metrics_table(metrics: list[PolicyMetrics]) -> str:
    lines = [
        "| policy | success_rate | invalid_call_rate | mean_ms | p95_ms | retries_per_success | cost_per_success_usd |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        cost = (
            _fmt_float(row.estimated_cost_per_successful_task_usd, 6)
            if row.estimated_cost_per_successful_task_usd is not None
            else "n/a"
        )
        lines.append(
            "| "
            f"{row.policy} | "
            f"{_fmt_float(row.task_success_rate)} | "
            f"{_fmt_float(row.invalid_tool_call_rate)} | "
            f"{_fmt_float(row.mean_latency_ms, 2)} | "
            f"{_fmt_float(row.p95_latency_ms, 2)} | "
            f"{_fmt_float(row.retries_per_successful_task, 3)} | "
            f"{cost} |"
        )
    return "\n".join(lines)


def failure_taxonomy(result: BenchmarkResult) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for task in result.task_results:
        if task.final_status != "success":
            counter[task.final_status] += 1
        for attempt in task.attempts:
            if attempt.status != "success":
                counter[attempt.status] += 1
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def render_markdown_summary(result: BenchmarkResult) -> str:
    taxonomy = failure_taxonomy(result)
    best_success = max(result.policy_metrics, key=lambda row: row.task_success_rate)
    lowest_invalid = min(result.policy_metrics, key=lambda row: row.invalid_tool_call_rate)
    lines = [
        "## Benchmark Summary",
        "",
        "### Verdict Snapshot",
        "",
        f"- best_success_policy: {best_success.policy} ({best_success.task_success_rate:.4f})",
        f"- lowest_invalid_policy: {lowest_invalid.policy} ({lowest_invalid.invalid_tool_call_rate:.4f})",
        "",
        policy_metrics_table(result.policy_metrics),
        "",
        "## Failure Taxonomy",
    ]
    if not taxonomy:
        lines.append("- no failures")
    else:
        for name, count in taxonomy:
            lines.append(f"- {name}: {count}")
    return "\n".join(lines) + "\n"


def write_markdown_summary(result: BenchmarkResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_summary(result), encoding="utf-8")


def _fmt_ci(stats: dict[str, Any], digits: int = 4) -> str:
    mean = float(stats.get("mean", 0.0))
    ci = float(stats.get("ci95_half_width", 0.0))
    return f"{mean:.{digits}f} +/- {ci:.{digits}f}"


def render_multi_seed_markdown(
    payload: dict,
    *,
    asset_paths: dict[str, str] | None = None,
) -> str:
    summary = summarize_multi_seed_payload(payload)
    lines = [
        "## Multi-Seed Aggregate",
        "",
        f"Seeds: {', '.join(str(seed) for seed in payload.get('seeds', []))}",
        "",
        "### Verdict Snapshot",
        "",
        f"- policy_count: {summary.get('policy_count', 0)}",
        f"- best_success_policy: {summary.get('best_success_policy', 'n/a')} ({float(summary.get('best_success_rate', 0.0)):.4f})",
        f"- lowest_invalid_policy: {summary.get('lowest_invalid_policy', 'n/a')} ({float(summary.get('lowest_invalid_rate', 0.0)):.4f})",
        "",
        "| policy | success_rate (mean+/-ci95) | invalid_call_rate (mean+/-ci95) | mean_ms (mean+/-ci95) | p95_ms (mean+/-ci95) | retries_per_success (mean+/-ci95) | cost_per_success_usd (mean+/-ci95) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in payload.get("aggregate_policy_metrics", []):
        policy = row.get("policy", "unknown")
        metrics = row.get("metrics", {})
        lines.append(
            "| "
            f"{policy} | "
            f"{_fmt_ci(metrics.get('task_success_rate', {}), 4)} | "
            f"{_fmt_ci(metrics.get('invalid_tool_call_rate', {}), 4)} | "
            f"{_fmt_ci(metrics.get('mean_latency_ms', {}), 2)} | "
            f"{_fmt_ci(metrics.get('p95_latency_ms', {}), 2)} | "
            f"{_fmt_ci(metrics.get('retries_per_successful_task', {}), 3)} | "
            f"{_fmt_ci(metrics.get('estimated_cost_per_successful_task_usd', {}), 6)} |"
        )

    _append_asset_section(lines, asset_paths)
    _append_asset_images(lines, asset_paths)
    return "\n".join(lines) + "\n"


def render_sweep_markdown(payload: dict) -> str:
    lines = [
        f"## Sweep: {payload.get('name', 'sweep')}",
    ]
    description = str(payload.get("description", "")).strip()
    if description:
        lines.extend(["", description])

    lines.extend(
        [
            "",
            "| scenario | policy | success_rate (mean+/-ci95) | p95_ms (mean+/-ci95) | cost_per_success_usd (mean+/-ci95) |",
            "|---|---|---:|---:|---:|",
        ]
    )

    for scenario in payload.get("scenarios", []):
        scenario_label = scenario.get("label", scenario.get("id", "scenario"))
        result = scenario.get("result", {})
        for row in result.get("aggregate_policy_metrics", []):
            metrics = row.get("metrics", {})
            lines.append(
                "| "
                f"{scenario_label} | "
                f"{row.get('policy', 'unknown')} | "
                f"{_fmt_ci(metrics.get('task_success_rate', {}), 4)} | "
                f"{_fmt_ci(metrics.get('p95_latency_ms', {}), 2)} | "
                f"{_fmt_ci(metrics.get('estimated_cost_per_successful_task_usd', {}), 6)} |"
            )

    return "\n".join(lines) + "\n"


def write_markdown_text(text: str, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_delta_markdown(
    payload: dict,
    *,
    asset_paths: dict[str, str] | None = None,
) -> str:
    summary = summarize_delta_payload(payload)
    lines = [
        "## Base vs Comparison Delta",
        "",
        "Delta is computed as comparison - base.",
        "",
        "### Verdict Snapshot",
        "",
        f"- compared_policies: {summary.get('policy_rows', 0)}",
        f"- mean_success_delta: {float(summary.get('mean_success_delta', 0.0)):+.4f}",
        f"- mean_invalid_delta: {float(summary.get('mean_invalid_delta', 0.0)):+.4f}",
        f"- best_success_policy: {summary.get('best_success_policy', 'n/a')} ({float(summary.get('best_success_delta', 0.0)):+.4f})",
        f"- max_abs_core_delta: {float(summary.get('max_abs_core_delta', 0.0)):.6f}",
        "",
    ]

    def _append_table(section_name: str, section_payload: dict) -> None:
        lines.extend(
            [
                f"### {section_name}",
                "",
                "| policy | delta_success_rate | delta_invalid_call_rate | delta_mean_ms | delta_p95_ms | delta_retries_per_success | delta_cost_per_success_usd |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        rows = section_payload.get("policies", [])
        if not rows:
            lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            lines.append("")
            return

        for row in rows:
            delta = row.get("delta", {})

            def _delta_value(name: str, digits: int) -> str:
                value = delta.get(name)
                if value is None:
                    return "n/a"
                return f"{float(value):.{digits}f}"

            lines.append(
                "| "
                f"{row.get('policy', 'unknown')} | "
                f"{_delta_value('task_success_rate', 4)} | "
                f"{_delta_value('invalid_tool_call_rate', 4)} | "
                f"{_delta_value('mean_latency_ms', 2)} | "
                f"{_delta_value('p95_latency_ms', 2)} | "
                f"{_delta_value('retries_per_successful_task', 3)} | "
                f"{_delta_value('estimated_cost_per_successful_task_usd', 6)} |"
            )
        lines.append("")

    target = payload.get("target")
    if isinstance(target, dict):
        _append_table("Target Workload", target)

    open_payload = payload.get("open")
    if isinstance(open_payload, dict):
        _append_table("Open Workload", open_payload)

    _append_asset_section(lines, asset_paths)
    _append_asset_images(lines, asset_paths)
    return "\n".join(lines).rstrip() + "\n"


def render_study_gate_markdown(
    payload: dict,
    *,
    asset_paths: dict[str, str] | None = None,
) -> str:
    lines = [
        "## Study Gate Report",
        "",
        f"Verdict: {payload.get('verdict', 'FAIL')}",
        "",
        "### Checks",
        "",
        "| check | status | value | threshold | detail |",
        "|---|---|---:|---:|---|",
    ]

    checks = list(payload.get("checks", []))
    if not checks:
        lines.append("| n/a | FAIL | n/a | n/a | no checks were evaluated |")
    else:
        for check in checks:
            status = "PASS" if bool(check.get("passed")) else "FAIL"
            lines.append(
                "| "
                f"{check.get('name', 'check')} | "
                f"{status} | "
                f"{check.get('value', 'n/a')} | "
                f"{check.get('threshold', 'n/a')} | "
                f"{check.get('detail', '')} |"
            )

    base_vs_comparison = payload.get("base_vs_comparison")
    if isinstance(base_vs_comparison, dict):
        lines.extend(
            [
                "",
                "### Base vs Comparison Signal",
                "",
                f"- policy_rows: {base_vs_comparison.get('policy_rows', 0)}",
                f"- mean_success_delta: {float(base_vs_comparison.get('mean_success_delta', 0.0)):+.4f}",
                f"- mean_invalid_delta: {float(base_vs_comparison.get('mean_invalid_delta', 0.0)):+.4f}",
                f"- max_abs_core_delta: {float(base_vs_comparison.get('max_abs_core_delta', 0.0)):.6f}",
            ]
        )

    null_summary = payload.get("null_control")
    if isinstance(null_summary, dict):
        lines.extend(
            [
                "",
                "### Null Control Signal",
                "",
                f"- policies_compared: {null_summary.get('policies_compared', 0)}",
                f"- mean_advantage_success_delta: {float(null_summary.get('mean_advantage_success_delta', 0.0)):+.4f}",
                f"- mean_advantage_invalid_delta: {float(null_summary.get('mean_advantage_invalid_delta', 0.0)):+.4f}",
                f"- max_abs_advantage_delta: {float(null_summary.get('max_abs_advantage_delta', 0.0)):.6f}",
            ]
        )

    matrix_summary = payload.get("matrix")
    if isinstance(matrix_summary, dict):
        lines.extend(
            [
                "",
                "### Transfer Matrix Signal",
                "",
                f"- rows_total: {matrix_summary.get('rows_total', 0)}",
                f"- portfolio_verdict: {matrix_summary.get('portfolio_verdict', '')}",
                f"- max_abs_delta: {float(matrix_summary.get('max_abs_delta', 0.0)):.6f}",
                f"- mean_abs_first_delta: {float(matrix_summary.get('mean_abs_first_delta', 0.0)):.6f}",
                f"- mean_abs_sequence_delta: {float(matrix_summary.get('mean_abs_sequence_delta', 0.0)):.6f}",
            ]
        )

    _append_asset_section(lines, asset_paths)
    _append_asset_images(lines, asset_paths)
    return "\n".join(lines).rstrip() + "\n"


def summarize_multi_seed_payload(payload: dict) -> dict[str, Any]:
    rows = list(payload.get("aggregate_policy_metrics", []))
    if not rows:
        return {
            "policy_count": 0,
            "best_success_policy": None,
            "lowest_invalid_policy": None,
        }

    def _mean(row: dict, metric_name: str) -> float:
        return float(row.get("metrics", {}).get(metric_name, {}).get("mean", 0.0) or 0.0)

    best_success = max(rows, key=lambda row: _mean(row, "task_success_rate"))
    lowest_invalid = min(rows, key=lambda row: _mean(row, "invalid_tool_call_rate"))
    return {
        "policy_count": len(rows),
        "best_success_policy": str(best_success.get("policy", "unknown")),
        "best_success_rate": _mean(best_success, "task_success_rate"),
        "lowest_invalid_policy": str(lowest_invalid.get("policy", "unknown")),
        "lowest_invalid_rate": _mean(lowest_invalid, "invalid_tool_call_rate"),
    }


def summarize_delta_payload(payload: dict) -> dict[str, Any]:
    target = payload.get("target", {})
    rows = list(target.get("policies", [])) if isinstance(target, dict) else []
    success_values = [
        float(row.get("delta", {}).get("task_success_rate"))
        for row in rows
        if row.get("delta", {}).get("task_success_rate") is not None
    ]
    invalid_values = [
        float(row.get("delta", {}).get("invalid_tool_call_rate"))
        for row in rows
        if row.get("delta", {}).get("invalid_tool_call_rate") is not None
    ]
    best_policy = None
    if rows:
        best_policy = max(
            rows,
            key=lambda row: float(row.get("delta", {}).get("task_success_rate", float("-inf")) or float("-inf")),
        )

    core_deltas = success_values + invalid_values
    return {
        "policy_rows": len(rows),
        "mean_success_delta": sum(success_values) / len(success_values) if success_values else 0.0,
        "mean_invalid_delta": sum(invalid_values) / len(invalid_values) if invalid_values else 0.0,
        "max_abs_core_delta": max((abs(value) for value in core_deltas), default=0.0),
        "best_success_policy": (
            str(best_policy.get("policy", "unknown")) if isinstance(best_policy, dict) else None
        ),
        "best_success_delta": (
            float(best_policy.get("delta", {}).get("task_success_rate", 0.0) or 0.0)
            if isinstance(best_policy, dict)
            else 0.0
        ),
    }


def summarize_transfer_matrix_payload(payload: dict) -> dict[str, Any]:
    rows = list(payload.get("rows", []))
    if not rows:
        return {
            "rows_total": 0,
            "portfolio_verdict": str(payload.get("portfolio_verdict", "")),
            "max_abs_delta": 0.0,
            "worst_toolset": None,
        }

    max_abs_delta = 0.0
    worst_row = None
    worst_score = float("inf")
    for row in rows:
        first = float(row.get("delta_first_tool_accuracy", 0.0) or 0.0)
        seq = float(row.get("delta_sequence_prefix_accuracy", 0.0) or 0.0)
        max_abs_delta = max(max_abs_delta, abs(first), abs(seq))
        row_score = min(first, seq)
        if row_score < worst_score:
            worst_score = row_score
            worst_row = row

    return {
        "rows_total": len(rows),
        "portfolio_verdict": str(payload.get("portfolio_verdict", "")),
        "max_abs_delta": max_abs_delta,
        "worst_toolset": (
            str(worst_row.get("toolset_id", "unknown")) if isinstance(worst_row, dict) else None
        ),
        "worst_first_delta": (
            float(worst_row.get("delta_first_tool_accuracy", 0.0) or 0.0)
            if isinstance(worst_row, dict)
            else 0.0
        ),
        "worst_sequence_delta": (
            float(worst_row.get("delta_sequence_prefix_accuracy", 0.0) or 0.0)
            if isinstance(worst_row, dict)
            else 0.0
        ),
    }


def render_analysis_markdown(
    *,
    multi_seed_payload: dict | None = None,
    delta_payload: dict | None = None,
    matrix_payload: dict | None = None,
    study_gate_payload: dict | None = None,
    source_paths: dict[str, str] | None = None,
    plot_paths: dict[str, str] | None = None,
) -> str:
    lines = [
        "# Run Analysis Summary",
        "",
        "Generated from checked-in benchmark artifacts.",
    ]

    if source_paths:
        lines.extend(["", "## Inputs", ""])
        for name, path in source_paths.items():
            lines.append(f"- {name}: `{path}`")

    lines.extend(["", "## Snapshot", ""])
    if study_gate_payload is not None:
        lines.append(f"- study_gate_verdict: {study_gate_payload.get('verdict', 'FAIL')}")
    if matrix_payload is not None:
        matrix_summary = summarize_transfer_matrix_payload(matrix_payload)
        lines.append(
            f"- matrix_portfolio_verdict: {matrix_summary.get('portfolio_verdict', '') or 'n/a'}"
        )
    if delta_payload is not None:
        delta_summary = summarize_delta_payload(delta_payload)
        lines.append(
            f"- mean_target_success_delta: {float(delta_summary.get('mean_success_delta', 0.0)):+.4f}"
        )
        lines.append(
            f"- mean_target_invalid_delta: {float(delta_summary.get('mean_invalid_delta', 0.0)):+.4f}"
        )
    if multi_seed_payload is None and delta_payload is None and matrix_payload is None and study_gate_payload is None:
        lines.append("- no supported payloads were provided")

    if multi_seed_payload is not None:
        summary = summarize_multi_seed_payload(multi_seed_payload)
        lines.extend(
            [
                "",
                "## Multi-Seed Overview",
                "",
                f"- policy_count: {summary.get('policy_count', 0)}",
                f"- best_success_policy: {summary.get('best_success_policy', 'n/a')} ({float(summary.get('best_success_rate', 0.0)):.4f})",
                f"- lowest_invalid_policy: {summary.get('lowest_invalid_policy', 'n/a')} ({float(summary.get('lowest_invalid_rate', 0.0)):.4f})",
            ]
        )
        if plot_paths and plot_paths.get("multi_seed"):
            lines.extend(["", f"![Multi-seed overview]({plot_paths['multi_seed']})"])

    if delta_payload is not None:
        summary = summarize_delta_payload(delta_payload)
        lines.extend(
            [
                "",
                "## Delta Overview",
                "",
                f"- compared_policies: {summary.get('policy_rows', 0)}",
                f"- mean_success_delta: {float(summary.get('mean_success_delta', 0.0)):+.4f}",
                f"- mean_invalid_delta: {float(summary.get('mean_invalid_delta', 0.0)):+.4f}",
                f"- best_success_policy: {summary.get('best_success_policy', 'n/a')} ({float(summary.get('best_success_delta', 0.0)):+.4f})",
            ]
        )
        if plot_paths and plot_paths.get("delta"):
            lines.extend(["", f"![Delta policy view]({plot_paths['delta']})"])

    if matrix_payload is not None:
        summary = summarize_transfer_matrix_payload(matrix_payload)
        lines.extend(
            [
                "",
                "## Transfer Matrix Overview",
                "",
                f"- portfolio_verdict: {summary.get('portfolio_verdict', 'n/a') or 'n/a'}",
                f"- rows_total: {summary.get('rows_total', 0)}",
                f"- max_abs_delta: {float(summary.get('max_abs_delta', 0.0)):.4f}",
                f"- worst_toolset: {summary.get('worst_toolset', 'n/a')} (first={float(summary.get('worst_first_delta', 0.0)):+.4f}, sequence={float(summary.get('worst_sequence_delta', 0.0)):+.4f})",
            ]
        )
        if plot_paths and plot_paths.get("matrix"):
            lines.extend(["", f"![Transfer matrix view]({plot_paths['matrix']})"])

    if study_gate_payload is not None:
        lines.extend(["", "## Study Gate", ""])
        for check in study_gate_payload.get("checks", []):
            status = "PASS" if bool(check.get("passed")) else "FAIL"
            lines.append(
                f"- {check.get('name', 'check')}: {status} (value={check.get('value', 'n/a')}, threshold={check.get('threshold', 'n/a')})"
            )

    return "\n".join(lines).rstrip() + "\n"
