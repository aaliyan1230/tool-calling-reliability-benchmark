from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .audit import audit_run, validate_outputs
from .airline_expansion import prepare_airline_seed_expansion, self_review_airline_seeds, select_airline_seeds
from .augmentation import audit_pilot, make_review_packet, run_pilot
from .augmentation_freeze import audit_augmented_dataset, freeze_augmented_dataset
from .hard import (
    audit_hard_dataset,
    audit_hard_run,
    freeze_hard_dataset,
    hard_config_context,
    make_hard_review_packet,
    select_hard_seeds,
)
from .seed_registry import select_fill_seeds, select_refill_seeds, select_retail_seeds
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
    matrix.add_argument("--profiles", default="safety_monitoring_v1,crm_handoff_v1,compact_crm_handoff_v1,plain_text_crm_handoff_v1")
    matrix.add_argument("--dataset", choices=["natural", "augmented"], default="natural")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--stage", choices=["smoke", "core", "stability"], required=True)
    summarize.add_argument("--providers", default="deepseek,gpt")
    summarize.add_argument("--profiles", default="safety_monitoring_v1,crm_handoff_v1,compact_crm_handoff_v1,plain_text_crm_handoff_v1")
    summarize.add_argument("--spend-cap-usd", type=float, default=25.0)
    summarize.add_argument("--build-views", action="store_true")
    summarize.add_argument("--dataset", choices=["natural", "augmented"], default="natural")
    validate = sub.add_parser("validate")
    validate.add_argument("--stage", choices=["smoke", "core", "stability"], default=None)
    audit = sub.add_parser("audit")
    audit.add_argument("--stage", choices=["smoke", "core", "stability"], default=None)
    augment = sub.add_parser("augment")
    augment.add_argument("--stage", choices=["pilot"], default="pilot")
    augment.add_argument("--spend-cap-usd", type=float, default=None)
    augment.add_argument("--seed-set", choices=["pilot", "scale", "fill", "refill"], default="pilot")
    audit_aug = sub.add_parser("audit-augmentation")
    audit_aug.add_argument("--seed-set", choices=["pilot", "scale", "fill", "refill"], default="pilot")
    review_aug = sub.add_parser("make-augmentation-review")
    review_aug.add_argument("--seed-set", choices=["pilot", "scale", "fill", "refill"], default="pilot")
    review_aug.add_argument("--supplement", action="store_true")
    review_aug.add_argument("--domain", choices=["airline", "retail"], default=None)
    sub.add_parser("select-retail-seeds")
    sub.add_parser("select-fill-seeds")
    sub.add_parser("select-refill-seeds")
    freeze_aug = sub.add_parser("freeze-augmented-dataset")
    freeze_aug.add_argument("--pairs-per-domain", type=int, default=15)
    sub.add_parser("audit-augmented-dataset")
    sub.add_parser("expand-airline-seeds")
    sub.add_parser("review-airline-seeds")
    sub.add_parser("select-airline-seeds")
    sub.add_parser("hard-preflight")
    sub.add_parser("audit-hard-source-pool")
    sub.add_parser("scan-hard-families")
    supplement_proposal = sub.add_parser("propose-hard-supplement")
    supplement_proposal.add_argument("--cases-per-domain", type=int, default=8)
    sub.add_parser("register-hard-supplement")
    wave_proposal = sub.add_parser("propose-hard-supplement-wave")
    wave_proposal.add_argument("--wave", type=int, default=2)
    wave_proposal.add_argument("--cases-per-domain-family", type=int, default=3)
    wave_register = sub.add_parser("register-hard-supplement-wave")
    wave_register.add_argument("--wave", type=int, default=2)
    wave_register.add_argument("--cases-per-domain-family", type=int, default=3)
    hard_seeds = sub.add_parser("select-hard-seeds")
    hard_seeds.add_argument("--stage", choices=["smoke", "core", "reserve", "supplement", "supplement2"], default="core")
    hard_aug = sub.add_parser("augment-hard")
    hard_aug.add_argument("--stage", choices=["smoke", "core", "reserve", "supplement", "supplement2"], required=True)
    hard_aug.add_argument("--spend-cap-usd", type=float, default=None)
    hard_audit = sub.add_parser("audit-hard")
    hard_audit.add_argument("--stage", choices=["smoke", "core", "reserve", "supplement", "supplement2"], default=None)
    hard_review = sub.add_parser("make-hard-review")
    hard_review.add_argument("--stage", choices=["smoke", "core", "reserve", "supplement", "supplement2"], required=True)
    seed_review = sub.add_parser("make-hard-seed-review")
    seed_review.add_argument("--limit-per-domain", type=int, default=40)
    seed_review.add_argument("--supplement", action="store_true")
    seed_review.add_argument("--domain", choices=["airline", "retail"], default="airline")
    merge_seed_review = sub.add_parser("merge-hard-seed-review")
    merge_seed_review.add_argument("--supplement", action="store_true")
    sub.add_parser("propose-hard-seed-mapping")
    register_seeds = sub.add_parser("register-hard-seeds")
    register_seeds.add_argument("--mapping", type=Path, required=True)
    sub.add_parser("freeze-hard")
    sub.add_parser("audit-hard-dataset")
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
        profiles = tuple(x.strip() for x in args.profiles.split(",") if x.strip())
        specs = call_matrix(args.local_root, args.run_root, args.stage, providers, args.dataset, profiles)
        result = {"stage": args.stage, **estimate_matrix(args.local_root, specs)}
        result["dataset"] = args.dataset
    elif command == "summarize":
        providers = tuple(x.strip() for x in args.providers.split(",") if x.strip())
        profiles = tuple(x.strip() for x in args.profiles.split(",") if x.strip())
        result = run_stage(args.local_root, args.run_root, args.stage, providers, args.spend_cap_usd, dataset=args.dataset, profiles=profiles)
        if args.build_views:
            result["views"] = build_views(args.local_root, args.run_root, args.stage, args.dataset, providers, profiles)
    elif command == "validate":
        result = validate_outputs(args.local_root, args.run_root, args.stage)
    elif command == "audit":
        result = audit_run(args.local_root, args.run_root, args.stage)
    elif command == "augment":
        result = run_pilot(args.local_root, args.run_root, args.spend_cap_usd, seed_set=args.seed_set)
    elif command == "audit-augmentation":
        result = audit_pilot(args.run_root, args.seed_set, args.local_root)
    elif command == "make-augmentation-review":
        result = make_review_packet(args.local_root, args.run_root, args.seed_set, args.supplement, args.domain)
    elif command == "select-retail-seeds":
        result = select_retail_seeds(args.local_root)
    elif command == "select-fill-seeds":
        result = select_fill_seeds(args.local_root)
    elif command == "select-refill-seeds":
        result = select_refill_seeds(args.local_root)
    elif command == "freeze-augmented-dataset":
        result = freeze_augmented_dataset(args.local_root, args.run_root, args.pairs_per_domain)
    elif command == "audit-augmented-dataset":
        result = audit_augmented_dataset(args.local_root)
    elif command == "expand-airline-seeds":
        result = prepare_airline_seed_expansion(args.local_root)
    elif command == "review-airline-seeds":
        result = self_review_airline_seeds(args.local_root)
    elif command == "select-airline-seeds":
        result = select_airline_seeds(args.local_root)
    elif command == "hard-preflight":
        stage_results = {}
        with hard_config_context():
            for stage in ("smoke", "core", "reserve"):
                try:
                    rows = select_hard_seeds(args.local_root, stage, args.run_root)
                    stage_results[stage] = {
                        "passed": True,
                        "rows": len(rows),
                        "by_domain": dict(Counter(row["domain"] for row in rows)),
                    }
                except ValueError as exc:
                    stage_results[stage] = {"passed": False, "errors": str(exc).splitlines()[1:]}
        result = {"version": "tcrb-hard-1", "stages": stage_results, "passed": all(item["passed"] for item in stage_results.values())}
    elif command == "audit-hard-source-pool":
        from .hard import audit_hard_source_pool

        result = audit_hard_source_pool(args.local_root, args.run_root)
    elif command == "scan-hard-families":
        from .hard_scan import scan_hard_families

        result = scan_hard_families(args.local_root, args.run_root)
    elif command == "propose-hard-supplement":
        from .hard_scan import propose_hard_supplement

        result = propose_hard_supplement(args.local_root, args.run_root, args.cases_per_domain)
    elif command == "register-hard-supplement":
        from .hard_scan import register_hard_supplement

        result = register_hard_supplement(args.local_root, args.run_root)
    elif command == "propose-hard-supplement-wave":
        from .hard_scan import propose_hard_supplement_wave

        result = propose_hard_supplement_wave(
            args.local_root,
            args.run_root,
            wave=args.wave,
            cases_per_domain_family=args.cases_per_domain_family,
        )
    elif command == "register-hard-supplement-wave":
        from .hard_scan import register_hard_supplement_wave

        result = register_hard_supplement_wave(
            args.local_root,
            args.run_root,
            wave=args.wave,
            cases_per_domain_family=args.cases_per_domain_family,
        )
    elif command == "select-hard-seeds":
        with hard_config_context():
            rows = select_hard_seeds(args.local_root, args.stage, args.run_root)
        result = {"version": "tcrb-hard-1", "stage": args.stage, "rows": len(rows), "case_ids": [row["case_id"] for row in rows], "passed": True}
    elif command == "augment-hard":
        with hard_config_context():
            result = run_pilot(args.local_root, args.run_root, args.spend_cap_usd, seed_set=f"hard_{args.stage}")
    elif command == "audit-hard":
        stages = [args.stage] if args.stage else ["smoke", "core", "reserve"]
        results = [audit_hard_run(args.local_root, args.run_root, stage) for stage in stages]
        result = {"version": "tcrb-hard-1", "stages": results, "passed": all(item["passed"] for item in results)}
    elif command == "make-hard-review":
        result = make_hard_review_packet(args.local_root, args.run_root, args.stage)
    elif command == "make-hard-seed-review":
        from .hard import make_hard_seed_review_packet, make_hard_seed_review_supplement

        if args.supplement:
            result = make_hard_seed_review_supplement(args.local_root, args.run_root, args.limit_per_domain, args.domain)
        else:
            result = make_hard_seed_review_packet(args.local_root, args.run_root, args.limit_per_domain)
    elif command == "merge-hard-seed-review":
        from .hard import merge_hard_seed_review

        result = merge_hard_seed_review(args.local_root, args.run_root, args.supplement)
    elif command == "propose-hard-seed-mapping":
        from .hard import propose_hard_seed_mapping

        result = propose_hard_seed_mapping(args.local_root, args.run_root)
    elif command == "register-hard-seeds":
        from .hard import register_hard_seed_mapping

        result = register_hard_seed_mapping(args.local_root, args.run_root, args.mapping)
    elif command == "freeze-hard":
        result = freeze_hard_dataset(args.local_root, args.run_root)
    elif command == "audit-hard-dataset":
        result = audit_hard_dataset(args.local_root)
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
