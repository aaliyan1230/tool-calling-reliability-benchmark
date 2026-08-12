from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .audit import audit_run, validate_outputs
from .augmentation import audit_pilot, make_review_packet, run_pilot
from .selection import build_candidates, freeze_dataset, make_annotation_packets, merge_annotations
from .sources import audit_sources, fetch_sources, normalize_sources
from .summaries import build_views, call_matrix, estimate_matrix, run_stage
from .util import DEFAULT_LOCAL_ROOT, DEFAULT_RUN_ROOT, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCRB v3.4 τ-bench summary dataset pipeline")
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--tau-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--primary-only", action="store_true")
    audit_source = sub.add_parser("audit-source")
    audit_source.add_argument("--primary-only", action="store_true")
    sub.add_parser("normalize")
    candidates = sub.add_parser("build-candidates")
    candidates.add_argument("--max-pairs-per-group", type=int, default=1)
    packets = sub.add_parser("make-annotations")
    packets.add_argument("--reviewers", default="reviewer_a,reviewer_b")
    packets.add_argument("--supplement", action="store_true")
    merge = sub.add_parser("merge-annotations")
    merge.add_argument("--reviewers", default="reviewer_a,reviewer_b")
    freeze = sub.add_parser("freeze-dataset")
    freeze.add_argument("--pairs-per-domain", type=int, default=15)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--stage", choices=["smoke", "core", "stability"], required=True)
    matrix.add_argument("--providers", default="deepseek,gpt")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--stage", choices=["smoke", "core", "stability"], required=True)
    summarize.add_argument("--providers", default="deepseek,gpt")
    summarize.add_argument("--spend-cap-usd", type=float, default=25.0)
    summarize.add_argument("--build-views", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--stage", choices=["smoke", "core", "stability"], default=None)
    audit = sub.add_parser("audit")
    audit.add_argument("--stage", choices=["smoke", "core", "stability"], default=None)
    augment = sub.add_parser("augment")
    augment.add_argument("--stage", choices=["pilot"], default="pilot")
    augment.add_argument("--spend-cap-usd", type=float, default=None)
    sub.add_parser("audit-augmentation")
    sub.add_parser("make-augmentation-review")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.local_root.mkdir(parents=True, exist_ok=True)
    command = args.command
    if command == "fetch":
        result = fetch_sources(args.local_root, include_fallback=not args.primary_only)
    elif command == "audit-source":
        tau_root = args.tau_root or default_tau_root()
        result = audit_sources(args.local_root, tau_root if tau_root.exists() else None, include_fallback=not args.primary_only)
    elif command == "normalize":
        result = normalize_sources(args.local_root, include_fallback=False)
    elif command == "build-candidates":
        result = build_candidates(args.local_root, max_pairs_per_group=args.max_pairs_per_group)
    elif command == "make-annotations":
        result = make_annotation_packets(args.local_root, tuple(x.strip() for x in args.reviewers.split(",") if x.strip()), supplement=args.supplement)
    elif command == "merge-annotations":
        result = merge_annotations(args.local_root, tuple(x.strip() for x in args.reviewers.split(",") if x.strip()))
    elif command == "freeze-dataset":
        result = freeze_dataset(args.local_root, args.pairs_per_domain)
    elif command == "matrix":
        providers = tuple(x.strip() for x in args.providers.split(",") if x.strip())
        specs = call_matrix(args.local_root, args.run_root, args.stage, providers)
        result = {"stage": args.stage, **estimate_matrix(args.local_root, specs)}
    elif command == "summarize":
        providers = tuple(x.strip() for x in args.providers.split(",") if x.strip())
        result = run_stage(args.local_root, args.run_root, args.stage, providers, args.spend_cap_usd)
        if args.build_views:
            result["views"] = build_views(args.local_root, args.run_root, args.stage)
    elif command == "validate":
        result = validate_outputs(args.local_root, args.run_root, args.stage)
    elif command == "audit":
        result = audit_run(args.local_root, args.run_root, args.stage)
    elif command == "augment":
        result = run_pilot(args.local_root, args.run_root, args.spend_cap_usd)
    elif command == "audit-augmentation":
        result = audit_pilot(args.run_root)
    elif command == "make-augmentation-review":
        result = make_review_packet(args.local_root, args.run_root)
    else:
        raise AssertionError(command)
    log_progress(command, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 1


def default_tau_root() -> Path:
    return Path(__file__).resolve().parents[3].parent / "tau2-snapshot"


def log_progress(command: str, result: dict) -> None:
    path = Path(__file__).resolve().parents[3] / "docs" / "v3" / "v3.4" / "progress-log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# v3.4 progress log\n\n", encoding="utf-8")
    timestamp = datetime.now(timezone.utc).isoformat()
    compact = json.dumps({key: value for key, value in result.items() if key not in {"artifact_sha256", "inventory"}}, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"- `{timestamp}` `{command}`: `{compact}`\n")
