from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a DPO preference dataset for first-tool routing repair"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--matrix-json", default=None)
    parser.add_argument("--toolsets", default=None)
    parser.add_argument("--max-alternatives", type=int, default=2)
    parser.add_argument("--base-hard-negative-repeat", type=int, default=1)
    parser.add_argument("--comparison-hard-negative-repeat", type=int, default=2)
    return parser.parse_args()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tool_payload(tool_name: str) -> str:
    payload = {
        "tool_calls": [
            {
                "name": str(tool_name).strip(),
                "arguments": {},
            }
        ]
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _selected_toolsets(args: argparse.Namespace) -> set[str] | None:
    if not args.toolsets:
        return None
    values = {token.strip() for token in args.toolsets.split(",") if token.strip()}
    return values or None


def _add_row(
    rows: list[dict[str, Any]],
    *,
    prompt: str,
    chosen_tool: str,
    rejected_tool: str,
    source: str,
    metadata: dict[str, Any],
    repeat: int = 1,
) -> None:
    if not prompt or not chosen_tool or not rejected_tool or chosen_tool == rejected_tool:
        return
    row = {
        "prompt": prompt,
        "chosen": _tool_payload(chosen_tool),
        "rejected": _tool_payload(rejected_tool),
        "source": source,
        "metadata": metadata,
    }
    for _ in range(max(1, repeat)):
        rows.append(dict(row))


def _eval_case_rows(
    *,
    manifest_rows: list[dict[str, Any]],
    selected_toolsets: set[str] | None,
    max_alternatives: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        toolset_id = str(manifest_row.get("toolset_id", "")).strip()
        if selected_toolsets and toolset_id not in selected_toolsets:
            continue
        eval_cases = _read_json(manifest_row["eval_cases"])
        for case in list(eval_cases.get("cases", [])):
            prompt = str(case.get("question", "")).strip()
            expected = str(case.get("expected_first_tool", "")).strip()
            alternatives = [
                str(name).strip()
                for name in list(case.get("acceptable_alternatives", []))[: max(0, max_alternatives)]
                if str(name).strip()
            ]
            for index, alternative in enumerate(alternatives, start=1):
                _add_row(
                    rows,
                    prompt=prompt,
                    chosen_tool=expected,
                    rejected_tool=alternative,
                    source="eval_case_alternative",
                    metadata={
                        "toolset_id": toolset_id,
                        "task_id": case.get("task_id"),
                        "negative_rank": index,
                    },
                )
    return rows


def _matrix_rows(
    *,
    matrix_json: str | None,
    selected_toolsets: set[str] | None,
    base_repeat: int,
    comparison_repeat: int,
) -> list[dict[str, Any]]:
    if not matrix_json:
        return []
    payload = _read_json(matrix_json)
    rows: list[dict[str, Any]] = []
    for toolset_row in list(payload.get("rows", [])):
        toolset_id = str(toolset_row.get("toolset_id", "")).strip()
        if selected_toolsets and toolset_id not in selected_toolsets:
            continue
        for planner_key, repeat in (
            ("base", max(0, base_repeat)),
            ("comparison", max(0, comparison_repeat)),
        ):
            planner_rows = list(toolset_row.get(planner_key, {}).get("rows", []))
            for case in planner_rows:
                prompt = str(case.get("question", "")).strip()
                expected = str(case.get("expected_first_tool", "")).strip()
                rejected = str(case.get("called_first_tool", "")).strip()
                if not expected or not rejected or expected == rejected:
                    continue
                _add_row(
                    rows,
                    prompt=prompt,
                    chosen_tool=expected,
                    rejected_tool=rejected,
                    source=f"choice_matrix_{planner_key}",
                    metadata={
                        "toolset_id": toolset_id,
                        "task_id": case.get("task_id"),
                        "planner_key": planner_key,
                    },
                    repeat=repeat,
                )
    return rows


def main() -> int:
    args = parse_args()
    manifest_rows = list(_read_json(args.manifest).get("toolsets", []))
    selected_toolsets = _selected_toolsets(args)

    rows = _eval_case_rows(
        manifest_rows=manifest_rows,
        selected_toolsets=selected_toolsets,
        max_alternatives=args.max_alternatives,
    )
    rows.extend(
        _matrix_rows(
            matrix_json=args.matrix_json,
            selected_toolsets=selected_toolsets,
            base_repeat=args.base_hard_negative_repeat,
            comparison_repeat=args.comparison_hard_negative_repeat,
        )
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=True) for row in rows]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
    print(
        json.dumps(
            {
                "rows": len(rows),
                "sources": source_counts,
                "output_jsonl": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
