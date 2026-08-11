from __future__ import annotations

import hashlib
import json
from typing import Any

# Imported rather than reimplemented: `full` and `actionable_summary` must be
# byte-identical to the v3.1 views or the replication claim is worthless.
from tcrb.v031.views import _actionable_summary_event, _event_dict

from .schema import CONTENT, PACKAGING, MonitorView, PrivateGold, RawTrace, ViewType


def _opaque_view_id(trace: RawTrace, view_type: str) -> str:
    # Salted with v032 so a v3.2 view id can never collide with a v3.1 one.
    value = f"{trace.trace_id}|{view_type}|v032"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _is_identifier(key: str) -> bool:
    """Same identifier rule v3.1 used, so both stripped views strip alike."""
    return key.endswith("_id") or key in {
        "id",
        "order_id",
        "transaction_id",
        "sku",
        "loan_id",
        "package",
    }


def _identifiers(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if _is_identifier(key)}


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _lossless_summary_event(trace: RawTrace) -> dict[str, Any]:
    """Summary packaging that loses nothing.

    Structurally this is `actionable_summary` with one addition: the
    answer-critical items keep their `payload`. Same summariser source, same
    item shape, same ordering. The presence of those values is the only thing
    that differs, which is what makes the pair a clean one-factor comparison.
    """
    required = set(trace.required_evidence_ids)
    items: list[dict[str, Any]] = []
    source_ids: list[str] = []
    for event in trace.events:
        if event.event_id == "subject_output":
            continue
        source_ids.append(event.event_id)
        if event.event_id in required:
            items.append(
                {
                    "event_id": event.event_id,
                    "source": event.source,
                    "operation": event.operation,
                    "event_type": event.event_type,
                    "observed": True,
                    "identifiers": _identifiers(event.payload),
                    "available_fields": sorted(event.payload),
                    # The one difference from actionable_summary.
                    "payload": event.payload,
                }
            )
        else:
            items.append(
                {
                    "event_id": event.event_id,
                    "source": event.source,
                    "operation": event.operation,
                    "event_type": event.event_type,
                    "payload": event.payload,
                }
            )
    payload = {
        "summary": items,
        "source_event_ids": source_ids,
        # Parallel in form to actionable_summary's "structured_field_projection".
        # Deliberately descriptive of the transform and not of its completeness:
        # a label saying "nothing is missing" would hand the monitor the answer.
        "compression_policy": "structured_field_passthrough",
    }
    return {
        "event_id": "lossless_summary",
        "step": 0,
        "source": "handoff_summarizer",
        "operation": "summarize",
        "payload": payload,
        "source_kind": "environment",
        "event_type": "handoff_summary",
        "integrity_hash": _hash(payload),
    }


def _raw_projection_events(trace: RawTrace) -> list[dict[str, Any]]:
    """Raw packaging that loses the same values `actionable_summary` loses.

    No summariser, no wrapper event, no compression policy. Ordinary trace
    events straight from the original sources — the answer-critical ones simply
    came back without their values, the way a partial tool response looks.
    """
    required = set(trace.required_evidence_ids)
    visible: list[dict[str, Any]] = []
    for event in trace.events:
        record = _event_dict(event)
        if event.event_id in required:
            projected = {
                **_identifiers(event.payload),
                "available_fields": sorted(event.payload),
            }
            record["payload"] = projected
            record["integrity_hash"] = _hash(projected)
        visible.append(record)
    return visible


def build_view(trace: RawTrace, view_type: ViewType) -> tuple[MonitorView, PrivateGold]:
    events = list(trace.events)
    if view_type == "full":
        visible = [_event_dict(event) for event in events]
    elif view_type == "lossless_summary":
        visible = [
            _event_dict(event) for event in events if event.event_id == "subject_output"
        ]
        visible.append(_lossless_summary_event(trace))
    elif view_type == "raw_projection":
        visible = _raw_projection_events(trace)
    elif view_type == "actionable_summary":
        visible = [
            _event_dict(event) for event in events if event.event_id == "subject_output"
        ]
        visible.append(_actionable_summary_event(trace))
    else:  # pragma: no cover - the Literal keeps callers out of here
        raise ValueError(f"unknown view type: {view_type}")

    view_id = _opaque_view_id(trace, view_type)
    view = MonitorView(
        view_id=view_id,
        trace_id=trace.trace_id,
        dataset=trace.dataset,
        domain=trace.domain,
        task_id=trace.task_id,
        task_family=trace.task_family,
        claim=trace.claim,
        view_type=view_type,  # type: ignore[arg-type]
        events=tuple(visible),
        completeness_manifest=None,
    )
    gold = PrivateGold(
        view_id=view_id,
        trace_id=trace.trace_id,
        dataset=trace.dataset,
        task_family=trace.task_family,
        gold_outcome=trace.gold_outcome,
        required_evidence_ids=trace.required_evidence_ids,
        expected_event_types=trace.expected_event_types,
        view_type=view_type,  # type: ignore[arg-type]
    )
    return view, gold


def build_all_views(
    traces: tuple[RawTrace, ...],
    view_types: tuple[ViewType, ...],
) -> tuple[tuple[MonitorView, PrivateGold], ...]:
    return tuple(
        build_view(trace, view_type) for trace in traces for view_type in view_types
    )


def _scalars(node: Any, *, key: str | None = None, into: set[str] | None = None) -> set[str]:
    """Every scalar reachable under `node`, as JSON.

    One walker is used for both sides of the comparison. Doing it twice with
    slightly different rules is how a list value ends up recorded whole on one
    side and element-by-element on the other, which silently breaks the check.

    Skipped: `integrity_hash` (long hex, matches anything), `available_fields`
    (field names, not values), identifiers (kept in every view by design), and
    bools (too little information to separate two views).
    """
    found = set() if into is None else into
    if key in {"integrity_hash", "available_fields"}:
        return found
    if isinstance(node, dict):
        for child_key, child in node.items():
            _scalars(child, key=child_key, into=found)
    elif isinstance(node, list):
        for child in node:
            _scalars(child, key=key, into=found)
    elif isinstance(node, bool) or node is None:
        return found
    elif not (key is not None and _is_identifier(key)):
        found.add(json.dumps(node, sort_keys=True))
    return found


def _payload_values(payload: dict[str, Any]) -> set[str]:
    return _scalars(payload)


def critical_values_by_event(trace: RawTrace) -> dict[str, set[str]]:
    """Per answer-critical event, the values a monitor would need from it.

    Deliberately *not* subtracting values the agent's own report also states.
    In every safe case the agent's number agrees with the independent record by
    construction — that is what makes it safe — so subtracting would wipe out
    most of the safe half of the dataset and, for `fi_transfer_acc100_acc200`,
    all of it. What matters is whether the *independent record's* value is
    readable, not whether that number appears somewhere else on screen.
    """
    required = set(trace.required_evidence_ids)
    return {
        event.event_id: _payload_values(event.payload)
        for event in trace.events
        if event.event_id in required
    }


def exposed_values_by_event(view: MonitorView) -> dict[str, set[str]]:
    """Per event, the values a monitor can actually read for it in this view.

    Looks up the record that stands for each event, whether that is a raw event
    or an item inside a summariser payload, and reads only that record. Reading
    the whole view instead would let one event's values be credited to another —
    which is exactly what happens when the agent's report repeats a number the
    independent record also holds.
    """
    exposed: dict[str, set[str]] = {}
    for event in view.events:
        if event.get("source") == "handoff_summarizer":
            for item in event.get("payload", {}).get("summary", []):
                if isinstance(item, dict) and isinstance(item.get("event_id"), str):
                    exposed[item["event_id"]] = _scalars(item.get("payload"))
        else:
            exposed[event["event_id"]] = _scalars(event.get("payload"))
    return exposed


def validate_views(
    traces: tuple[RawTrace, ...],
    pairs: tuple[tuple[MonitorView, PrivateGold], ...],
) -> list[str]:
    """The checks that make the 2x2 trustworthy.

    The important ones are the last two: values-present views must actually
    carry every answer-critical value, and values-stripped views must actually
    carry none of them. If those slip, the factor labels are lies and the whole
    run means nothing.
    """
    errors: list[str] = []
    raw_by_id = {trace.trace_id: trace for trace in traces}
    seen: set[str] = set()

    for view, gold in pairs:
        if view.view_id in seen:
            errors.append(f"duplicate view id {view.view_id}")
        seen.add(view.view_id)
        raw = raw_by_id.get(view.trace_id)
        if raw is None:
            errors.append(f"{view.view_id}: missing raw trace")
            continue
        if gold.view_type != view.view_type:
            errors.append(f"{view.view_id}: public/private view type mismatch")

        public = view.to_public_dict()
        forbidden = {
            "gold_outcome",
            "required_evidence_ids",
            "mutation_id",
            "faults_applied",
            "diagnostic_labels",
            "payload_state",
        }
        leaked = forbidden & _recursive_keys(public)
        if leaked:
            errors.append(f"{view.view_id}: private key leaked: {sorted(leaked)}")

        needed = critical_values_by_event(raw)
        exposed = exposed_values_by_event(view)

        for event_id, values in needed.items():
            seen = exposed.get(event_id, set())
            if CONTENT[view.view_type] == "values_present":
                missing = values - seen
                if missing:
                    errors.append(
                        f"{view.view_id} ({view.view_type}): {event_id} is missing "
                        f"values {sorted(missing)[:4]}"
                    )
            elif seen:
                errors.append(
                    f"{view.view_id} ({view.view_type}): {event_id} still exposes "
                    f"values {sorted(seen)[:4]}"
                )

        if PACKAGING[view.view_type] == "summary":
            wrappers = [
                event
                for event in view.events
                if event.get("source") == "handoff_summarizer"
            ]
            if len(wrappers) != 1:
                errors.append(f"{view.view_id}: expected exactly one summary event")
        else:
            if any(event.get("source") == "handoff_summarizer" for event in view.events):
                errors.append(f"{view.view_id}: raw view must not contain a summary event")

    by_trace: dict[str, set[str]] = {}
    for view, _ in pairs:
        by_trace.setdefault(view.trace_id, set()).add(view.view_type)
    expected = set(PACKAGING) & {view.view_type for view, _ in pairs}
    for trace_id, types in by_trace.items():
        if types != expected:
            errors.append(f"{trace_id}: unbalanced cell set {sorted(types)}")

    return errors


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(_recursive_keys(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(_recursive_keys(child))
        return result
    return set()
