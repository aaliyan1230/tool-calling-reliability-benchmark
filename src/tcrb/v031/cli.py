from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_run
from .audit import audit_run
from .reporting import build_report
from .runner import build_call_specs, prepare_dataset, run_stage
from .schema import FOLLOWUP_VIEW_TYPES, VIEW_TYPES


DEFAULT_RUN_DIR = Path("outputs/v031_visibility_pilot/pilot_01")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCRB v3.1 visibility pilot")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--view-set", choices=["primary", "actionable"], default="primary")
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--stage", choices=["smoke", "core", "stability"], required=True)
    matrix.add_argument("--providers", default="deepseek,gpt")
    run = sub.add_parser("run")
    run.add_argument("--stage", choices=["smoke", "core", "stability"], required=True)
    run.add_argument("--providers", default="deepseek,gpt")
    run.add_argument("--spend-cap-usd", type=float, default=25.0)
    run.add_argument("--timeout-s", type=int, default=120)
    run.add_argument("--max-retries", type=int, default=4)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--bootstrap-samples", type=int, default=4000)
    sub.add_parser("audit")
    sub.add_parser("report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        view_types = FOLLOWUP_VIEW_TYPES if args.view_set == "actionable" else VIEW_TYPES
        print(json.dumps(prepare_dataset(args.run_dir, view_types=view_types), indent=2))
        return 0
    if args.command == "matrix":
        providers = tuple(item.strip() for item in args.providers.split(",") if item.strip())
        specs = build_call_specs(args.run_dir, stage=args.stage, providers=providers)
        print(json.dumps({"stage": args.stage, "calls": len(specs)}, indent=2))
        return 0
    if args.command == "run":
        providers = tuple(item.strip() for item in args.providers.split(",") if item.strip())
        result = run_stage(
            args.run_dir,
            stage=args.stage,
            providers=providers,
            spend_cap_usd=args.spend_cap_usd,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
        )
        print(json.dumps(result, indent=2))
        return 1 if result["failed_now"] else 0
    if args.command == "analyze":
        print(json.dumps(analyze_run(args.run_dir, bootstrap_samples=args.bootstrap_samples), indent=2))
        return 0
    if args.command == "audit":
        result = audit_run(args.run_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    if args.command == "report":
        print(json.dumps(build_report(args.run_dir), indent=2))
        return 0
    raise AssertionError("unreachable")
