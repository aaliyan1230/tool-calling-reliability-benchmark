from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .models import BenchmarkResult, PolicyMetrics


def _fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


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
    lines = [
        "## Benchmark Summary",
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


def render_multi_seed_markdown(payload: dict) -> str:
    lines = [
        "## Multi-Seed Aggregate",
        "",
        f"Seeds: {', '.join(str(seed) for seed in payload.get('seeds', []))}",
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


def render_delta_markdown(payload: dict) -> str:
    lines = [
        "## Base vs Finetuned Delta",
        "",
        "Delta is computed as finetuned - base.",
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

    return "\n".join(lines).rstrip() + "\n"
