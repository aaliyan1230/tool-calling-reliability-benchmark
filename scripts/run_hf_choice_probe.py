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
        description="Directly compare first-tool choices for two HF planners"
    )
    parser.add_argument("--workload", required=True)
    parser.add_argument("--eval-cases-json", required=True)
    parser.add_argument("--base-planner-config", required=True)
    parser.add_argument("--comparison-planner-config", required=True)
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--policy", default="naive_retry")
    parser.add_argument("--output-dir", default="runs/hf-choice-probe")
    return parser.parse_args()


def _case_map(eval_cases: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id", "")): row for row in eval_cases.get("cases", [])}


def _run_planner(
    *,
    planner_config: str,
    workload_path: str,
    eval_cases_path: str,
    max_tasks: int,
    policy: str,
) -> dict[str, Any]:
    workload = load_workload(workload_path)
    eval_cases = load_eval_cases(eval_cases_path)
    cases_by_task = _case_map(eval_cases)
    planner = load_tool_planner(planner_config)

    rows: list[dict[str, Any]] = []
    tasks = workload.tasks[: max(0, max_tasks)] if max_tasks > 0 else workload.tasks
    for index, task in enumerate(tasks, start=1):
        case = cases_by_task.get(task.task_id, {})
        expected = str(case.get("expected_first_tool", ""))
        print(
            f"[choice-probe] {planner.planner_id} {index}/{len(tasks)} {task.task_id}",
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
        "planner_config": planner_config,
        "cases_total": len(rows),
        "first_tool_accuracy": (correct / len(rows)) if rows else 0.0,
        "rows": rows,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    base = payload["base"]
    comparison = payload["comparison"]
    delta = payload["delta_first_tool_accuracy"]
    lines = [
        "# HF Choice Probe",
        "",
        f"- base_planner: {base['planner_id']}",
        f"- comparison_planner: {comparison['planner_id']}",
        f"- cases_total: {base['cases_total']}",
        f"- base_first_tool_accuracy: {base['first_tool_accuracy']:.4f}",
        f"- comparison_first_tool_accuracy: {comparison['first_tool_accuracy']:.4f}",
        f"- delta_first_tool_accuracy: {delta:+.4f}",
        "",
        "| task | expected | base | comparison | changed |",
        "|---|---|---|---|---|",
    ]
    for base_row, comparison_row in zip(base["rows"], comparison["rows"], strict=True):
        changed = base_row["called_first_tool"] != comparison_row["called_first_tool"]
        lines.append(
            "| {task} | {expected} | {base_tool} | {comparison_tool} | {changed} |".format(
                task=base_row["task_id"],
                expected=base_row["expected_first_tool"],
                base_tool=base_row["called_first_tool"],
                comparison_tool=comparison_row["called_first_tool"],
                changed="yes" if changed else "no",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    load_env_file(Path.cwd())
    args = parse_args()
    base = _run_planner(
        planner_config=args.base_planner_config,
        workload_path=args.workload,
        eval_cases_path=args.eval_cases_json,
        max_tasks=args.max_tasks,
        policy=args.policy,
    )
    comparison = _run_planner(
        planner_config=args.comparison_planner_config,
        workload_path=args.workload,
        eval_cases_path=args.eval_cases_json,
        max_tasks=args.max_tasks,
        policy=args.policy,
    )
    payload = {
        "base": base,
        "comparison": comparison,
        "delta_first_tool_accuracy": comparison["first_tool_accuracy"]
        - base["first_tool_accuracy"],
        "choice_changes": sum(
            1
            for base_row, comparison_row in zip(base["rows"], comparison["rows"], strict=True)
            if base_row["called_first_tool"] != comparison_row["called_first_tool"]
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "choice_probe.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "choice_probe.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    print(f"[choice-probe] Wrote {output_dir / 'choice_probe.json'}", flush=True)
    print(f"[choice-probe] Wrote {output_dir / 'choice_probe.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
