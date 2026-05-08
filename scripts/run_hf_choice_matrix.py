from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tcrb.config import load_workload
from tcrb.env import load_env_file
from tcrb.eval_cases import load_eval_cases
from tcrb.planner import load_tool_planner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly compare first-tool choices across a workload manifest"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-planner-config", required=True)
    parser.add_argument("--comparison-planner-config", required=True)
    parser.add_argument("--toolsets", default=None)
    parser.add_argument("--max-tasks", type=int, default=18)
    parser.add_argument("--policy", default="naive_retry")
    parser.add_argument("--output-dir", default="runs/hf-choice-matrix")
    return parser.parse_args()


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = list(payload.get("toolsets", []))
    if not rows:
        raise ValueError(f"manifest has no toolsets: {path}")
    return rows


def _case_map(eval_cases: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id", "")): row for row in eval_cases.get("cases", [])}


def _score_planner(
    *,
    planner: Any,
    workload_path: str,
    eval_cases_path: str,
    max_tasks: int,
    policy: str,
    toolset_id: str,
) -> dict[str, Any]:
    workload = load_workload(workload_path)
    eval_cases = load_eval_cases(eval_cases_path)
    cases_by_task = _case_map(eval_cases)

    rows: list[dict[str, Any]] = []
    tasks = workload.tasks[: max(0, max_tasks)] if max_tasks > 0 else workload.tasks
    for index, task in enumerate(tasks, start=1):
        case = cases_by_task.get(task.task_id, {})
        expected = str(case.get("expected_first_tool", ""))
        print(
            f"[choice-matrix] {planner.planner_id} {toolset_id} {index}/{len(tasks)} {task.task_id}",
            flush=True,
        )
        called = planner.choose_tool(
            task=task,
            workload=workload,
            policy=policy,
            attempt_number=1,
            attempted_tools=set(),
            last_status=None,
            rng=None,  # type: ignore[arg-type]
        )
        rows.append(
            {
                "task_id": task.task_id,
                "question": case.get("question") or task.user_query,
                "expected_first_tool": expected,
                "called_first_tool": called,
                "correct": bool(expected) and called == expected,
            }
        )

    correct = sum(1 for row in rows if row["correct"])
    return {
        "planner_id": getattr(planner, "planner_id", "planner"),
        "cases_total": len(rows),
        "first_tool_accuracy": (correct / len(rows)) if rows else 0.0,
        "rows": rows,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# HF Choice Matrix",
        "",
        f"- base_planner: {payload['base_planner_id']}",
        f"- comparison_planner: {payload['comparison_planner_id']}",
        "",
        "| toolset | cases | base_acc | comparison_acc | delta | choice_changes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {toolset} | {cases} | {base:.4f} | {comparison:.4f} | {delta:+.4f} | {changes} |".format(
                toolset=row["toolset_id"],
                cases=row["cases_total"],
                base=row["base_first_tool_accuracy"],
                comparison=row["comparison_first_tool_accuracy"],
                delta=row["delta_first_tool_accuracy"],
                changes=row["choice_changes"],
            )
        )
    lines.extend(
        [
            "",
            "## Changed Choices",
            "",
            "| toolset | task | expected | base | comparison |",
            "|---|---|---|---|---|",
        ]
    )
    for row in payload["rows"]:
        for base_row, comparison_row in zip(
            row["base"]["rows"],
            row["comparison"]["rows"],
            strict=True,
        ):
            if base_row["called_first_tool"] == comparison_row["called_first_tool"]:
                continue
            lines.append(
                "| {toolset} | {task} | {expected} | {base_tool} | {comparison_tool} |".format(
                    toolset=row["toolset_id"],
                    task=base_row["task_id"],
                    expected=base_row["expected_first_tool"],
                    base_tool=base_row["called_first_tool"],
                    comparison_tool=comparison_row["called_first_tool"],
                )
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    load_env_file(Path.cwd())
    args = parse_args()
    manifest_rows = _read_manifest(args.manifest)
    if args.toolsets:
        selected = {token.strip() for token in args.toolsets.split(",") if token.strip()}
        manifest_rows = [
            row for row in manifest_rows if str(row.get("toolset_id", "")) in selected
        ]
        if not manifest_rows:
            raise ValueError("--toolsets filter removed all manifest rows")

    base_planner = load_tool_planner(args.base_planner_config)
    comparison_planner = load_tool_planner(args.comparison_planner_config)

    matrix_rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        toolset_id = str(manifest_row["toolset_id"])
        workload_path = str(manifest_row["workload"])
        eval_cases_path = str(manifest_row["eval_cases"])
        base = _score_planner(
            planner=base_planner,
            workload_path=workload_path,
            eval_cases_path=eval_cases_path,
            max_tasks=args.max_tasks,
            policy=args.policy,
            toolset_id=toolset_id,
        )
        comparison = _score_planner(
            planner=comparison_planner,
            workload_path=workload_path,
            eval_cases_path=eval_cases_path,
            max_tasks=args.max_tasks,
            policy=args.policy,
            toolset_id=toolset_id,
        )
        choice_changes = sum(
            1
            for base_row, comparison_row in zip(base["rows"], comparison["rows"], strict=True)
            if base_row["called_first_tool"] != comparison_row["called_first_tool"]
        )
        matrix_rows.append(
            {
                "toolset_id": toolset_id,
                "cases_total": base["cases_total"],
                "base_first_tool_accuracy": base["first_tool_accuracy"],
                "comparison_first_tool_accuracy": comparison["first_tool_accuracy"],
                "delta_first_tool_accuracy": comparison["first_tool_accuracy"]
                - base["first_tool_accuracy"],
                "choice_changes": choice_changes,
                "base": base,
                "comparison": comparison,
            }
        )

    payload = {
        "base_planner_id": getattr(base_planner, "planner_id", "base"),
        "comparison_planner_id": getattr(comparison_planner, "planner_id", "comparison"),
        "rows": matrix_rows,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "choice_matrix.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "choice_matrix.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    print(f"[choice-matrix] Wrote {output_dir / 'choice_matrix.json'}", flush=True)
    print(f"[choice-matrix] Wrote {output_dir / 'choice_matrix.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
