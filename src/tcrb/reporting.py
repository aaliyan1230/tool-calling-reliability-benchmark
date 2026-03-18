from __future__ import annotations

from collections import Counter
from pathlib import Path

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
