from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .benchmark import run_benchmark, write_result_json
from .config import load_benchmark_config, load_workload
from .reporting import write_markdown_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tool-calling reliability benchmark")
    parser.add_argument(
        "--config",
        default="configs/baseline.json",
        help="Path to benchmark config JSON",
    )
    parser.add_argument(
        "--workload",
        default="workloads/sample_tasks.json",
        help="Path to workload JSON",
    )
    parser.add_argument("--outdir", default="runs", help="Directory for run outputs")
    parser.add_argument("--label", default=None, help="Optional run label")
    return parser


def _default_label() -> str:
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{now}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    workload = load_workload(args.workload)
    config = load_benchmark_config(args.config)
    result = run_benchmark(workload=workload, config=config)

    label = args.label or _default_label()
    run_dir = Path(args.outdir) / label
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_json = run_dir / "result.json"
    summary_md = run_dir / "summary.md"

    write_result_json(result, raw_json)
    write_markdown_summary(result, summary_md)

    print(f"Wrote benchmark results: {raw_json}")
    print(f"Wrote markdown summary: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
