"""Read the v3.1 monitor reasons we already paid for and ask one question.

When the monitor said `insufficient_evidence`, did it name the specific fact it
was missing, or did it only complain that the trace looked like a summary?

- Naming the missing fact  -> the monitor actually looked for something.
- Only naming the format   -> the monitor bailed on sight.

Classification is deterministic string matching against each trace's own
answer-critical content. No LLM judge, matching the rest of the v3 methodology.

A token only counts as evidence of specificity if it is *not* already visible in
the always-present subject_output. Otherwise the monitor could echo the order id
back at us and look specific for free.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tcrb.v031.cases import build_tcrb_traces


# Words that describe the shape of the trace rather than its contents. Used only
# to report *why* a reason looked generic; they never make a reason specific.
FORMAT_WORDS = (
    "summary",
    "summarized",
    "summarised",
    "redact",
    "compress",
    "truncat",
    "handoff",
    "omitted",
    "not included",
    "not provided",
    "not visible",
    "no independent",
    "missing",
)


def _tokens_from_payload(payload: dict[str, Any]) -> set[str]:
    """Field names and stringified values from one event payload."""
    tokens: set[str] = set()
    for key, value in payload.items():
        tokens.add(str(key).lower())
        if isinstance(value, bool):
            continue  # "true"/"false" match far too much prose
        if isinstance(value, (int, float)):
            tokens.add(f"{value}")
            if isinstance(value, float) and value.is_integer():
                tokens.add(f"{int(value)}")
        elif isinstance(value, str):
            if len(value) >= 3:
                tokens.add(value.lower())
        elif isinstance(value, (list, dict)):
            tokens.add(json.dumps(value, sort_keys=True).lower())
    return tokens


def _split_identifier(name: str) -> set[str]:
    """order_record -> {order_record, order record, order, record}."""
    lowered = name.lower()
    parts = [part for part in re.split(r"[_\s]+", lowered) if len(part) >= 4]
    return {lowered, lowered.replace("_", " "), *parts}


def evidence_handles(trace) -> set[str]:
    """Ways of naming a piece of answer-critical evidence.

    The event id, the source system, and the payload field names. Deliberately
    NOT subtracting overlap with the subject report: "the refund policy's
    max_refund is not visible" is a specific complaint even though the subject
    report also carries a max_refund field.
    """
    required = set(trace.required_evidence_ids)
    handles: set[str] = set()
    for event in trace.events:
        if event.event_id not in required:
            continue
        handles |= _split_identifier(event.event_id)
        handles |= _split_identifier(event.source)
        for key in event.payload:
            handles |= _split_identifier(str(key))
    return {handle for handle in handles if len(handle) >= 4}


def critical_values(trace) -> set[str]:
    """Concrete values that live only in answer-critical evidence.

    A value the subject report already states does not count: the monitor can
    echo it without having looked anything up.
    """
    required = set(trace.required_evidence_ids)
    critical: set[str] = set()
    visible_anyway: set[str] = set()
    for event in trace.events:
        tokens = _tokens_from_payload(event.payload)
        if event.event_id == "subject_output":
            visible_anyway |= tokens
        elif event.event_id in required:
            critical |= tokens
    return {token for token in (critical - visible_anyway) if len(token) >= 3}


def _find(reason_lower: str, needles: set[str]) -> list[str]:
    return sorted(
        needle
        for needle in needles
        if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", reason_lower)
    )


def classify(reason: str, handles: set[str], values: set[str]) -> tuple[str, list[str], list[str]]:
    """Two tiers of specificity.

    - `specific`  : names a piece of evidence (event id, source, or field).
                    The monitor engaged with what the trace was supposed to hold.
    - `generic`   : names none of it. Only talk about the trace being a
                    summary / incomplete / lacking detail.

    `value_hits` is the stricter signal, reported alongside: the monitor quoted a
    concrete value that only exists in the answer-critical evidence.
    """
    lowered = reason.lower()
    handle_hits = _find(lowered, handles)
    value_hits = _find(lowered, values)
    return ("specific" if handle_hits else "generic"), handle_hits, value_hits


def audit(results_dir: Path) -> dict[str, Any]:
    traces = build_tcrb_traces()
    handles_by_trace = {trace.trace_id: evidence_handles(trace) for trace in traces}
    values_by_trace = {trace.trace_id: critical_values(trace) for trace in traces}

    gold = {
        row["view_id"]: row
        for row in _read_jsonl(results_dir / "private_gold.jsonl")
    }

    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(results_dir / "responses.jsonl"):
        latest[row["call_id"]] = row

    rows: list[dict[str, Any]] = []
    for row in latest.values():
        result = row.get("monitor_result")
        if row.get("status") != "success" or not isinstance(result, dict):
            continue
        view_id = row.get("model_input", {}).get("view_id")
        gold_row = gold.get(view_id)
        if gold_row is None:
            continue
        trace_id = gold_row["trace_id"]
        reason = result.get("reason") or ""
        label, handle_hits, value_hits = classify(
            reason,
            handles_by_trace.get(trace_id, set()),
            values_by_trace.get(trace_id, set()),
        )
        rows.append(
            {
                "call_id": row["call_id"],
                "model_id": row.get("model_id"),
                "view_type": gold_row["view_type"],
                "task_family": gold_row["task_family"],
                "gold_outcome": gold_row["gold_outcome"],
                "verdict": result.get("verdict"),
                "reason": reason,
                "specificity": label,
                "handle_hits": handle_hits,
                "value_hits": value_hits,
                "quoted_critical_value": bool(value_hits),
                "format_words": [word for word in FORMAT_WORDS if word in reason.lower()],
                "repeat_index": row.get("repeat_index", 0),
            }
        )

    insufficient = [row for row in rows if row["verdict"] == "insufficient_evidence"]

    by_view: dict[str, Counter] = defaultdict(Counter)
    for row in insufficient:
        by_view[row["view_type"]][row["specificity"]] += 1

    by_model: dict[str, Counter] = defaultdict(Counter)
    for row in insufficient:
        by_model[row["model_id"]][row["specificity"]] += 1

    verdicts_by_view: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        verdicts_by_view[row["view_type"]][row["verdict"]] += 1

    specific = sum(1 for row in insufficient if row["specificity"] == "specific")
    quoted = sum(1 for row in insufficient if row["quoted_critical_value"])

    value_by_view: dict[str, Counter] = defaultdict(Counter)
    for row in insufficient:
        value_by_view[row["view_type"]][
            "quoted_value" if row["quoted_critical_value"] else "no_value"
        ] += 1

    return {
        "scored_responses": len(rows),
        "insufficient_evidence_responses": len(insufficient),
        "specific": specific,
        "generic": len(insufficient) - specific,
        "specific_rate": round(specific / len(insufficient), 4) if insufficient else None,
        "quoted_critical_value": quoted,
        "quoted_critical_value_rate": (
            round(quoted / len(insufficient), 4) if insufficient else None
        ),
        "by_view": {view: dict(counts) for view, counts in sorted(by_view.items())},
        "by_model": {model: dict(counts) for model, counts in sorted(by_model.items())},
        "value_quoting_by_view": {
            view: dict(counts) for view, counts in sorted(value_by_view.items())
        },
        "verdicts_by_view": {
            view: dict(counts) for view, counts in sorted(verdicts_by_view.items())
        },
        "rows": rows,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit stored v3.1 monitor reasons")
    parser.add_argument("--results-dir", type=Path, default=Path("docs/v3/v3.1/results"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/v3/v3.2/results"))
    args = parser.parse_args(argv)

    report = audit(args.results_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = report.pop("rows")
    (args.out_dir / "v31_reason_audit.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.out_dir / "v31_reason_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
