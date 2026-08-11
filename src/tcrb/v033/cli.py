from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_run
from .audit import audit_run
from .reporting import build_report
from .runner import build_call_specs, prepare_dataset, run_stage
from .schema import VIEW_TYPES


DEFAULT_RUN_DIR = Path("outputs/v033_silent_vs_advertised/pilot_01")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TCRB v3.3 — scope vs announcement 2x2"
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare")

    matrix = sub.add_parser("matrix")
    matrix.add_argument("--stage", choices=["smoke", "core", "stability"], required=True)
    matrix.add_argument("--providers", default="deepseek,gpt")

    run = sub.add_parser("run")
    run.add_argument("--stage", choices=["smoke", "core", "stability"], required=True)
    run.add_argument("--providers", default="deepseek,gpt")
    run.add_argument("--spend-cap-usd", type=float, default=5.0)
    run.add_argument("--timeout-s", type=int, default=120)
    run.add_argument("--max-retries", type=int, default=4)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--bootstrap-samples", type=int, default=10000)

    sub.add_parser("audit")

    report = sub.add_parser("report")
    report.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prepare":
        print(json.dumps(prepare_dataset(args.run_dir, view_types=VIEW_TYPES), indent=2))
        return 0

    if args.command == "matrix":
        providers = _providers(args.providers)
        specs = build_call_specs(args.run_dir, stage=args.stage, providers=providers)
        print(json.dumps({"stage": args.stage, "calls": len(specs)}, indent=2))
        return 0

    if args.command == "run":
        result = run_stage(
            args.run_dir,
            stage=args.stage,
            providers=_providers(args.providers),
            spend_cap_usd=args.spend_cap_usd,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
        )
        print(json.dumps(result, indent=2))
        return 1 if result["failed_now"] else 0

    if args.command == "analyze":
        summary = analyze_run(args.run_dir, bootstrap_samples=args.bootstrap_samples)
        print(
            json.dumps(
                {k: v for k, v in summary.items() if k != "scored_rows"}, indent=2
            )
        )
        return 0

    if args.command == "audit":
        result = audit_run(args.run_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1

    if args.command == "report":
        print(json.dumps(build_report(args.run_dir, output_dir=args.output_dir), indent=2))
        return 0

    raise AssertionError("unreachable")


def _providers(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())
