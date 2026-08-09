from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_run
from .cases import build_case_variants, validate_case_variants
from .runner import build_call_specs, prepare_dataset, run_stage


DEFAULT_RUN_DIR = Path("outputs/v03_provenance_pilot/pilot_01")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCRB v0.3 evidence-provenance pilot")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Validate the fixed 2x2 dataset")
    prepare = subparsers.add_parser("prepare", help="Write public views and private gold")
    prepare.add_argument("--include-stress", action="store_true")

    matrix = subparsers.add_parser("matrix", help="Show scheduled call counts")
    matrix.add_argument("--stage", choices=["smoke", "core", "stability", "stress"], required=True)
    matrix.add_argument("--providers", default="deepseek,gpt")

    run = subparsers.add_parser("run", help="Run one resumable pilot stage")
    run.add_argument("--stage", choices=["smoke", "core", "stability", "stress"], required=True)
    run.add_argument("--providers", default="deepseek,gpt")
    run.add_argument("--spend-cap-usd", type=float, default=15.0)
    run.add_argument("--timeout-s", type=int, default=120)
    run.add_argument("--max-retries", type=int, default=4)

    analyze = subparsers.add_parser("analyze", help="Score completed responses")
    analyze.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        variants = build_case_variants()
        errors = validate_case_variants(variants)
        payload = {"base_cases": len(variants) // 4, "variants": len(variants), "errors": errors}
        print(json.dumps(payload, indent=2))
        return 1 if errors else 0
    if args.command == "prepare":
        print(json.dumps(prepare_dataset(args.run_dir, include_stress=args.include_stress), indent=2))
        return 0
    if args.command == "matrix":
        providers = tuple(item.strip() for item in args.providers.split(",") if item.strip())
        specs = build_call_specs(stage=args.stage, providers=providers)
        print(json.dumps({"stage": args.stage, "providers": providers, "calls": len(specs)}, indent=2))
        return 0
    if args.command == "run":
        providers = tuple(item.strip() for item in args.providers.split(",") if item.strip())
        summary = run_stage(
            args.run_dir,
            stage=args.stage,
            providers=providers,
            spend_cap_usd=args.spend_cap_usd,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
        )
        print(json.dumps(summary, indent=2))
        return 1 if summary["failed_now"] else 0
    if args.command == "analyze":
        print(json.dumps(analyze_run(args.run_dir, bootstrap_samples=args.bootstrap_samples), indent=2))
        return 0
    raise AssertionError("unreachable")
