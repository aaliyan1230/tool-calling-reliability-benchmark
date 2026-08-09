"""Repeat τ-bench monitor cells to check response stability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tcrb.v031.providers import call_monitor, parse_monitor_result
from tcrb.v031.schema import MonitorView


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--task-family", action="append", default=[])
    parser.add_argument("--view-type", action="append", default=[])
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run_dir / "verifier_views.jsonl").read_text().splitlines() if line.strip()]
    gold_rows = [
        json.loads(line)
        for line in (args.run_dir / "private_gold.jsonl").read_text().splitlines()
        if line.strip()
    ]
    gold = {row["view_id"]: row for row in gold_rows}
    for row in rows:
        row["_view_type"] = gold.get(row["view_id"], {}).get("view_type")
    if args.task_family:
        rows = [row for row in rows if row["task_family"] in set(args.task_family)]
    if args.view_type:
        rows = [row for row in rows if row.get("_view_type") in set(args.view_type)]
    output = args.run_dir / "stability_repeats.jsonl"
    records = []
    for provider in ("deepseek", "gpt"):
        for row in rows:
            view = MonitorView(
                view_id=row["view_id"],
                trace_id=row["trace_id"],
                dataset=row["dataset"],
                domain=row["domain"],
                task_id=row["task_id"],
                task_family=row["task_family"],
                claim=row["claim"],
                view_type="full",
                events=tuple(row["events"]),
                completeness_manifest=row.get("completeness_manifest"),
            )
            for repeat in range(1, args.repeats + 1):
                print(provider, row["task_family"], row["_view_type"], repeat, flush=True)
                parsed = None
                error = None
                response = None
                for attempt in range(4):
                    response = call_monitor(provider, view, timeout_s=120, max_retries=3)
                    try:
                        parsed = parse_monitor_result(response.output_text).to_dict()
                        error = None
                        break
                    except ValueError as exc:
                        error = str(exc)
                        if attempt < 3:
                            print(f"  parse retry {attempt + 1}: {error}", flush=True)
                assert response is not None
                records.append(
                    {
                        "repeat_id": hashlib.sha256(f"{provider}|{row['view_id']}|{repeat}".encode()).hexdigest()[:24],
                        "provider": provider,
                        "view_id": row["view_id"],
                        "task_family": row["task_family"],
                        "view_id": row["view_id"],
                        "repeat_index": repeat,
                        "monitor_result": parsed,
                        "parse_error": error,
                        "output_text": response.output_text,
                        "usage": response.usage,
                        "estimated_extra_cost_usd": response.estimated_extra_cost_usd,
                    }
                )
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    print(json.dumps({"records": len(records), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
