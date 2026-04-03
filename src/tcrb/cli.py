from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .benchmark import run_benchmark, write_result_json
from .config import load_benchmark_config, load_workload
from .experiments import parse_seed_list, run_multi_seed, run_sweep, write_json
from .eval_cases import load_eval_cases, score_eval_cases
from .finetune.dataset import build_examples_from_result_payload, split_examples, write_jsonl
from .finetune.evaluate import compare_run_payloads, load_json_payload
from .planner import load_tool_planner
from .reporting import (
    render_delta_markdown,
    render_multi_seed_markdown,
    render_study_gate_markdown,
    render_sweep_markdown,
    write_markdown_summary,
    write_markdown_text,
)
from .study_gate import StudyGateThresholds, evaluate_study_gates


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

    finetune_data_parser = subparsers.add_parser(
        "finetune-data", help="Build finetuning JSONL data from benchmark result JSON"
    )
    finetune_data_parser.add_argument(
        "--input-json",
        required=True,
        help="Path to runs/<label>/result.json",
    )
    finetune_data_parser.add_argument(
        "--output-dir",
        default="finetuned-models/training",
        help="Directory where train/eval JSONL files are written",
    )
    finetune_data_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation set fraction in [0.0, 1.0)",
    )
    finetune_data_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for train/eval split",
    )
    finetune_data_parser.add_argument(
        "--workload",
        default=None,
        help="Optional workload JSON used to enrich prompt context",
    )
    finetune_data_parser.add_argument(
        "--include-failure-attempts",
        action="store_true",
        help="Include non-success attempts as labels (invalid tool calls are always excluded)",
    )

    eval_delta_parser = subparsers.add_parser(
        "eval-delta", help="Compare base vs finetuned run payloads"
    )
    eval_delta_parser.add_argument(
        "--base-run",
        required=True,
        help="Path to base result JSON (result.json or multi_seed.json)",
    )
    eval_delta_parser.add_argument(
        "--finetuned-run",
        required=True,
        help="Path to finetuned result JSON (result.json or multi_seed.json)",
    )
    eval_delta_parser.add_argument(
        "--open-base-run",
        default=None,
        help="Optional base run JSON for held-out open workload",
    )
    eval_delta_parser.add_argument(
        "--open-finetuned-run",
        default=None,
        help="Optional finetuned run JSON for held-out open workload",
    )
    eval_delta_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path for delta JSON",
    )
    eval_delta_parser.add_argument(
        "--output-report",
        default=None,
        help="Optional output path for markdown report (defaults next to output JSON)",
    )

    study_gate_parser = subparsers.add_parser(
        "study-gate", help="Run anti-flatline study gate checks"
    )
    study_gate_parser.add_argument(
        "--base-run",
        required=True,
        help="Path to base result JSON (result.json or multi_seed.json)",
    )
    study_gate_parser.add_argument(
        "--finetuned-run",
        required=True,
        help="Path to finetuned result JSON (result.json or multi_seed.json)",
    )
    study_gate_parser.add_argument(
        "--null-run",
        default=None,
        help="Optional null-control run JSON for comparator checks",
    )
    study_gate_parser.add_argument(
        "--matrix-json",
        default=None,
        help="Optional transfer matrix JSON for matrix-level checks",
    )
    study_gate_parser.add_argument(
        "--flatline-epsilon",
        type=float,
        default=1e-4,
        help="Minimum absolute delta to avoid flatline verdict",
    )
    study_gate_parser.add_argument(
        "--min-effect-vs-null",
        type=float,
        default=3e-3,
        help="Minimum max abs effect vs null-control delta",
    )
    study_gate_parser.add_argument(
        "--matrix-flatline-epsilon",
        type=float,
        default=1e-4,
        help="Minimum transfer-matrix delta when matrix signal is required",
    )
    study_gate_parser.add_argument(
        "--require-matrix-signal",
        action="store_true",
        help="Require transfer-matrix deltas to be non-flat",
    )
    study_gate_parser.add_argument(
        "--require-matrix-not-fail",
        action="store_true",
        help="Require transfer-matrix portfolio verdict to not be FAIL",
    )
    study_gate_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path for study gate JSON",
    )
    study_gate_parser.add_argument(
        "--output-report",
        default=None,
        help="Optional output path for markdown summary",
    )
    study_gate_parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Return exit code 1 if any gate fails",
    )

    eval_cases_parser = subparsers.add_parser(
        "eval-cases-score",
        help="Score run output against question-level expected-tool eval cases",
    )
    eval_cases_parser.add_argument(
        "--result-json",
        required=True,
        help="Path to run result.json payload",
    )
    eval_cases_parser.add_argument(
        "--eval-cases-json",
        required=True,
        help="Path to eval cases JSON with expected tools",
    )
    eval_cases_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path for scored eval-case JSON",
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


def _run_finetune_data(args: argparse.Namespace) -> int:
    payload = load_json_payload(args.input_json)
    workload = load_workload(args.workload) if args.workload else None

    examples = build_examples_from_result_payload(
        payload,
        workload=workload,
        include_failure_attempts=bool(args.include_failure_attempts),
    )
    train_rows, eval_rows = split_examples(
        examples,
        validation_split=float(args.validation_split),
        seed=int(args.seed),
    )

    output_dir = Path(args.output_dir)
    train_path = output_dir / "train_dataset.jsonl"
    eval_path = output_dir / "eval_dataset.jsonl"

    write_jsonl(train_rows, train_path)
    write_jsonl(eval_rows, eval_path)

    print(f"Examples total: {len(examples)}")
    print(f"Train examples: {len(train_rows)}")
    print(f"Eval examples: {len(eval_rows)}")
    print(f"Wrote train dataset: {train_path}")
    print(f"Wrote eval dataset: {eval_path}")
    return 0


def _run_eval_delta(args: argparse.Namespace) -> int:
    base_payload = load_json_payload(args.base_run)
    finetuned_payload = load_json_payload(args.finetuned_run)

    report_payload: dict = {
        "target": compare_run_payloads(base_payload, finetuned_payload),
    }

    has_open_pair = args.open_base_run and args.open_finetuned_run
    if has_open_pair:
        open_base_payload = load_json_payload(args.open_base_run)
        open_finetuned_payload = load_json_payload(args.open_finetuned_run)
        report_payload["open"] = compare_run_payloads(
            open_base_payload, open_finetuned_payload
        )

    output_json = Path(args.output_json) if args.output_json else None
    if output_json is not None:
        write_json(report_payload, output_json)
        print(f"Wrote delta JSON: {output_json}")

    output_report = Path(args.output_report) if args.output_report else None
    if output_report is None and output_json is not None:
        output_report = output_json.with_suffix(".md")
    if output_report is not None:
        write_markdown_text(render_delta_markdown(report_payload), output_report)
        print(f"Wrote delta report: {output_report}")

    if output_json is None and output_report is None:
        print(json.dumps(report_payload, indent=2))

    return 0


def _run_study_gate(args: argparse.Namespace) -> int:
    base_payload = load_json_payload(args.base_run)
    finetuned_payload = load_json_payload(args.finetuned_run)
    null_payload = load_json_payload(args.null_run) if args.null_run else None
    matrix_payload = load_json_payload(args.matrix_json) if args.matrix_json else None

    thresholds = StudyGateThresholds(
        flatline_epsilon=float(args.flatline_epsilon),
        min_effect_vs_null=float(args.min_effect_vs_null),
        matrix_flatline_epsilon=float(args.matrix_flatline_epsilon),
        require_matrix_signal=bool(args.require_matrix_signal),
        require_matrix_not_fail=bool(args.require_matrix_not_fail),
    )

    report_payload = evaluate_study_gates(
        base_payload=base_payload,
        finetuned_payload=finetuned_payload,
        null_payload=null_payload,
        matrix_payload=matrix_payload,
        thresholds=thresholds,
    )

    output_json = Path(args.output_json) if args.output_json else None
    if output_json is not None:
        write_json(report_payload, output_json)
        print(f"Wrote study gate JSON: {output_json}")

    output_report = Path(args.output_report) if args.output_report else None
    if output_report is None and output_json is not None:
        output_report = output_json.with_suffix(".md")
    if output_report is not None:
        write_markdown_text(render_study_gate_markdown(report_payload), output_report)
        print(f"Wrote study gate markdown: {output_report}")

    print(f"Study gate verdict: {report_payload.get('verdict', 'FAIL')}")
    for check in report_payload.get("checks", []):
        status = "PASS" if bool(check.get("passed")) else "FAIL"
        value = check.get("value")
        threshold = check.get("threshold")
        print(
            f"- {check.get('name', 'check')}: {status} "
            f"(value={value}, threshold={threshold})"
        )

    if bool(args.fail_on_violation) and str(report_payload.get("verdict")) != "PASS":
        return 1
    return 0


def _run_eval_cases_score(args: argparse.Namespace) -> int:
    result_payload = load_json_payload(args.result_json)
    eval_cases_payload = load_eval_cases(args.eval_cases_json)
    scored = score_eval_cases(result_payload, eval_cases_payload)

    output_json = Path(args.output_json) if args.output_json else None
    if output_json is not None:
        write_json(scored, output_json)
        print(f"Wrote eval-case score JSON: {output_json}")
    else:
        print(json.dumps(scored, indent=2))

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
    if args.command == "finetune-data":
        return _run_finetune_data(args)
    if args.command == "eval-delta":
        return _run_eval_delta(args)
    if args.command == "study-gate":
        return _run_study_gate(args)
    if args.command == "eval-cases-score":
        return _run_eval_cases_score(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
