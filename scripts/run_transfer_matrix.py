from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcrb.benchmark import run_benchmark, write_result_json
from tcrb.config import load_benchmark_config, load_workload
from tcrb.env import load_env_file
from tcrb.eval_cases import load_eval_cases, score_eval_cases, write_json as write_json_file
from tcrb.models import Workload
from tcrb.planner import load_tool_planner
from tcrb.reporting import write_markdown_summary, write_markdown_text
from tcrb.transfer_matrix import (
    MatrixThresholds,
    gate_eval_case_delta,
    render_transfer_matrix_markdown,
    summarize_eval_case_score,
)
from tcrb.visualization import write_transfer_matrix_plot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run toolset transfer matrix with eval-case gate checks"
    )
    parser.add_argument(
        "--manifest",
        default="workloads/enriched/manifest.json",
        help="Path to enriched workload manifest",
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.json",
        help="Benchmark config path",
    )
    parser.add_argument(
        "--base-planner-config",
        default="configs/planners/policy_native.json",
        help="Planner config for baseline model/planner",
    )
    parser.add_argument(
        "--comparison-planner-config",
        required=True,
        help="Planner config for comparison model/planner",
    )
    parser.add_argument(
        "--target-toolset",
        required=True,
        help="Toolset id treated as target domain",
    )
    parser.add_argument(
        "--label",
        default="transfer-matrix",
        help="Output run label under runs/",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Reuse existing result files if present",
    )
    parser.add_argument(
        "--target-first-min-delta",
        type=float,
        default=0.03,
        help="Gate threshold for target first-tool accuracy delta",
    )
    parser.add_argument(
        "--target-seq-min-delta",
        type=float,
        default=0.03,
        help="Gate threshold for target sequence-prefix accuracy delta",
    )
    parser.add_argument(
        "--open-first-min-delta",
        type=float,
        default=-0.03,
        help="Gate threshold for open first-tool accuracy delta",
    )
    parser.add_argument(
        "--open-seq-min-delta",
        type=float,
        default=-0.03,
        help="Gate threshold for open sequence-prefix accuracy delta",
    )
    parser.add_argument(
        "--toolsets",
        default=None,
        help="Optional comma-separated toolset ids to include from manifest",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Optional max number of tasks per toolset (0 means all tasks)",
    )
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("toolsets", []))
    if not rows:
        raise ValueError(f"manifest has no toolsets: {path}")
    return rows


def _run_or_load_result(
    *,
    workload_path: Path,
    workload: Workload | None,
    config_path: Path,
    planner_config_path: Path,
    result_path: Path,
    summary_path: Path,
    skip_benchmark: bool,
) -> dict:
    if skip_benchmark and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    active_workload = workload if workload is not None else load_workload(workload_path)
    config = load_benchmark_config(config_path)
    planner = load_tool_planner(str(planner_config_path))
    result = run_benchmark(workload=active_workload, config=config, planner=planner)

    write_result_json(result, result_path)
    write_markdown_summary(result, summary_path)
    return json.loads(result_path.read_text(encoding="utf-8"))


def _maybe_limit_workload(
    workload: Workload,
    *,
    max_tasks: int,
) -> Workload:
    if max_tasks <= 0 or len(workload.tasks) <= max_tasks:
        return workload
    return Workload(tools=workload.tools, tasks=list(workload.tasks[:max_tasks]))


def _filter_eval_cases_for_task_ids(eval_payload: dict, task_ids: set[str]) -> dict:
    rows = [
        row
        for row in list(eval_payload.get("cases", []))
        if str(row.get("task_id", "")) in task_ids
    ]
    payload = dict(eval_payload)
    payload["cases"] = rows
    return payload


def main() -> int:
    args = _parse_args()
    repo_root = Path.cwd()
    load_env_file(repo_root)

    manifest_path = (repo_root / args.manifest).resolve()
    config_path = (repo_root / args.config).resolve()
    base_planner_config_path = (repo_root / args.base_planner_config).resolve()
    comparison_planner_config_path = (repo_root / args.comparison_planner_config).resolve()

    entries = _read_manifest(manifest_path)
    selected_toolsets = None
    if args.toolsets:
        selected_toolsets = {
            token.strip()
            for token in str(args.toolsets).split(",")
            if token.strip()
        }
        entries = [
            row for row in entries if str(row.get("toolset_id", "")) in selected_toolsets
        ]
        if not entries:
            raise ValueError("--toolsets filter removed all manifest entries")

    target_toolset = str(args.target_toolset)
    known_toolsets = {str(row.get("toolset_id", "")) for row in entries}
    if target_toolset not in known_toolsets:
        raise ValueError(f"target toolset '{target_toolset}' not found in manifest")

    thresholds = MatrixThresholds(
        target_first_tool_min_delta=float(args.target_first_min_delta),
        target_sequence_min_delta=float(args.target_seq_min_delta),
        open_first_tool_min_delta=float(args.open_first_min_delta),
        open_sequence_min_delta=float(args.open_seq_min_delta),
    )

    run_root = repo_root / "runs" / args.label
    run_root.mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict] = []
    for row in entries:
        toolset_id = str(row.get("toolset_id", ""))
        workload_path = (repo_root / str(row.get("workload", ""))).resolve()
        eval_cases_path = (repo_root / str(row.get("eval_cases", ""))).resolve()

        toolset_dir = run_root / toolset_id
        toolset_dir.mkdir(parents=True, exist_ok=True)

        base_result_path = toolset_dir / "base.result.json"
        base_summary_path = toolset_dir / "base.summary.md"
        comparison_result_path = toolset_dir / "comparison.result.json"
        comparison_summary_path = toolset_dir / "comparison.summary.md"

        workload_for_scores = load_workload(workload_path)
        workload_for_scores = _maybe_limit_workload(
            workload_for_scores,
            max_tasks=max(0, int(args.max_tasks)),
        )
        limited_task_ids = {task.task_id for task in workload_for_scores.tasks}

        base_result_payload = _run_or_load_result(
            workload_path=workload_path,
            workload=workload_for_scores,
            config_path=config_path,
            planner_config_path=base_planner_config_path,
            result_path=base_result_path,
            summary_path=base_summary_path,
            skip_benchmark=bool(args.skip_benchmark),
        )
        comparison_result_payload = _run_or_load_result(
            workload_path=workload_path,
            workload=workload_for_scores,
            config_path=config_path,
            planner_config_path=comparison_planner_config_path,
            result_path=comparison_result_path,
            summary_path=comparison_summary_path,
            skip_benchmark=bool(args.skip_benchmark),
        )

        eval_cases_payload = load_eval_cases(eval_cases_path)
        if int(args.max_tasks) > 0:
            eval_cases_payload = _filter_eval_cases_for_task_ids(
                eval_cases_payload,
                task_ids=limited_task_ids,
            )
        base_score = score_eval_cases(base_result_payload, eval_cases_payload)
        comparison_score = score_eval_cases(comparison_result_payload, eval_cases_payload)

        base_score_path = toolset_dir / "base.eval_case_score.json"
        comparison_score_path = toolset_dir / "comparison.eval_case_score.json"
        write_json_file(base_score, base_score_path)
        write_json_file(comparison_score, comparison_score_path)

        base_summary = summarize_eval_case_score(base_score)
        comparison_summary = summarize_eval_case_score(comparison_score)

        delta_first = float(comparison_summary.first_tool_accuracy - base_summary.first_tool_accuracy)
        delta_seq = float(
            comparison_summary.sequence_prefix_accuracy - base_summary.sequence_prefix_accuracy
        )

        split = "target" if toolset_id == target_toolset else "open"
        verdict = gate_eval_case_delta(
            split=split,
            first_tool_delta=delta_first,
            sequence_delta=delta_seq,
            thresholds=thresholds,
        )

        matrix_rows.append(
            {
                "toolset_id": toolset_id,
                "split": split,
                "cases_total": int(base_summary.cases_total),
                "base_first_tool_accuracy": float(base_summary.first_tool_accuracy),
                "comparison_first_tool_accuracy": float(comparison_summary.first_tool_accuracy),
                "delta_first_tool_accuracy": delta_first,
                "base_sequence_prefix_accuracy": float(base_summary.sequence_prefix_accuracy),
                "comparison_sequence_prefix_accuracy": float(
                    comparison_summary.sequence_prefix_accuracy
                ),
                "delta_sequence_prefix_accuracy": delta_seq,
                "verdict": verdict,
                "artifacts": {
                    "base_result": str(base_result_path.relative_to(repo_root)),
                    "comparison_result": str(comparison_result_path.relative_to(repo_root)),
                    "base_eval_case_score": str(base_score_path.relative_to(repo_root)),
                    "comparison_eval_case_score": str(comparison_score_path.relative_to(repo_root)),
                },
            }
        )

    portfolio = "PASS"
    if any(row["verdict"] == "FAIL" for row in matrix_rows):
        portfolio = "FAIL"
    elif any(row["verdict"] == "HOLD" for row in matrix_rows):
        portfolio = "HOLD"

    output_payload = {
        "type": "transfer_matrix",
        "target_toolset": target_toolset,
        "manifest": str(manifest_path.relative_to(repo_root)),
        "base_planner_config": str(base_planner_config_path.relative_to(repo_root)),
        "comparison_planner_config": str(comparison_planner_config_path.relative_to(repo_root)),
        "thresholds": {
            "target_first_tool_min_delta": thresholds.target_first_tool_min_delta,
            "target_sequence_min_delta": thresholds.target_sequence_min_delta,
            "open_first_tool_min_delta": thresholds.open_first_tool_min_delta,
            "open_sequence_min_delta": thresholds.open_sequence_min_delta,
        },
        "rows": matrix_rows,
        "portfolio_verdict": portfolio,
    }

    matrix_json_path = run_root / "matrix.json"
    matrix_md_path = run_root / "matrix_summary.md"
    matrix_plot_path = run_root / "transfer_matrix.png"
    write_json_file(output_payload, matrix_json_path)

    asset_paths: dict[str, str] | None = None
    try:
        write_transfer_matrix_plot(output_payload, matrix_plot_path)
        asset_paths = {"matrix_plot": matrix_plot_path.name}
        print(f"Wrote transfer matrix plot: {matrix_plot_path}")
    except RuntimeError:
        asset_paths = None

    write_markdown_text(
        render_transfer_matrix_markdown(
            target_toolset_id=target_toolset,
            rows=matrix_rows,
            thresholds=thresholds,
            asset_paths=asset_paths,
        ),
        matrix_md_path,
    )

    print(f"Wrote transfer matrix JSON: {matrix_json_path}")
    print(f"Wrote transfer matrix summary: {matrix_md_path}")
    print(f"Portfolio verdict: {portfolio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
