from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .benchmark import run_benchmark, write_result_json
from .config import load_benchmark_config, load_workload
from .experiments import parse_seed_list, run_multi_seed, run_sweep, write_json
from .planner import load_tool_planner
from .reporting import (
    render_multi_seed_markdown,
    render_sweep_markdown,
    write_markdown_summary,
    write_markdown_text,
)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument(
        "--planner-config",
        default=None,
        help="Optional tool planner JSON config",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tool-calling reliability benchmark")
    _add_run_args(parser)
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run one benchmark seed")
    _add_run_args(run_parser)

    multi_parser = subparsers.add_parser(
        "multi-seed", help="Run benchmark across many seeds"
    )
    multi_parser.add_argument(
        "--config",
        default="configs/baseline.json",
        help="Path to benchmark config JSON",
    )
    multi_parser.add_argument(
        "--workload",
        default="workloads/sample_tasks.json",
        help="Path to workload JSON",
    )
    multi_parser.add_argument(
        "--seeds", default="1,2,3,4,5", help="Comma-separated integer seeds"
    )
    multi_parser.add_argument(
        "--outdir", default="runs", help="Directory for run outputs"
    )
    multi_parser.add_argument("--label", default=None, help="Optional run label")
    multi_parser.add_argument(
        "--planner-config",
        default=None,
        help="Optional tool planner JSON config",
    )

    sweep_parser = subparsers.add_parser(
        "sweep", help="Run scenario sweep with multi-seed aggregation"
    )
    sweep_parser.add_argument(
        "--base-config",
        default="configs/baseline.json",
        help="Base benchmark config JSON",
    )
    sweep_parser.add_argument(
        "--sweep-config",
        default="configs/sweeps/fault_levels.json",
        help="Sweep definition JSON",
    )
    sweep_parser.add_argument(
        "--workload",
        default="workloads/sample_tasks.json",
        help="Path to workload JSON",
    )
    sweep_parser.add_argument(
        "--outdir", default="runs", help="Directory for run outputs"
    )
    sweep_parser.add_argument("--label", default=None, help="Optional run label")
    sweep_parser.add_argument(
        "--planner-config",
        default=None,
        help="Optional tool planner JSON config",
    )

    return parser


def _default_label() -> str:
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{now}"


def _load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_single(args: argparse.Namespace) -> int:
    workload = load_workload(args.workload)
    config = load_benchmark_config(args.config)
    planner = load_tool_planner(args.planner_config)
    result = run_benchmark(workload=workload, config=config, planner=planner)

    label = args.label or _default_label()
    run_dir = Path(args.outdir) / label
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_json = run_dir / "result.json"
    summary_md = run_dir / "summary.md"

    write_result_json(result, raw_json)
    write_markdown_summary(result, summary_md)

    print(f"Planner: {planner.planner_id}")
    print(f"Wrote benchmark results: {raw_json}")
    print(f"Wrote markdown summary: {summary_md}")
    return 0


def _run_multi_seed(args: argparse.Namespace) -> int:
    workload = load_workload(args.workload)
    config = load_benchmark_config(args.config)
    planner = load_tool_planner(args.planner_config)
    seeds = parse_seed_list(args.seeds)
    payload = run_multi_seed(
        workload=workload, config=config, seeds=seeds, planner=planner
    )

    label = args.label or _default_label()
    run_dir = Path(args.outdir) / label
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_json = run_dir / "multi_seed.json"
    summary_md = run_dir / "multi_seed_summary.md"

    write_json(payload, raw_json)
    write_markdown_text(render_multi_seed_markdown(payload), summary_md)

    print(f"Planner: {planner.planner_id}")
    print(f"Wrote multi-seed results: {raw_json}")
    print(f"Wrote multi-seed summary: {summary_md}")
    return 0


def _run_sweep(args: argparse.Namespace) -> int:
    workload = load_workload(args.workload)
    base_payload = _load_json(args.base_config)
    sweep_payload = _load_json(args.sweep_config)
    planner = load_tool_planner(args.planner_config)
    result = run_sweep(
        workload=workload,
        base_config_payload=base_payload,
        sweep_payload=sweep_payload,
        planner=planner,
    )

    label = args.label or _default_label()
    run_dir = Path(args.outdir) / label
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_json = run_dir / "sweep.json"
    summary_md = run_dir / "sweep_summary.md"

    write_json(result, raw_json)
    write_markdown_text(render_sweep_markdown(result), summary_md)

    print(f"Planner: {planner.planner_id}")
    print(f"Wrote sweep results: {raw_json}")
    print(f"Wrote sweep summary: {summary_md}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "run"):
        return _run_single(args)
    if args.command == "multi-seed":
        return _run_multi_seed(args)
    if args.command == "sweep":
        return _run_sweep(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
