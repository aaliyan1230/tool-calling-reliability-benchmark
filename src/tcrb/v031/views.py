from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from .schema import MonitorView, PrivateGold, RawTrace, TraceEvent, ViewType


def _opaque_view_id(trace: RawTrace, view_type: ViewType) -> str:
    value = f"{trace.trace_id}|{view_type}|v031"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _event_dict(event: TraceEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "step": event.step,
        "source": event.source,
        "operation": event.operation,
        "payload": event.payload,
        "source_kind": event.source_kind,
        "event_type": event.event_type,
        "integrity_hash": event.integrity_hash,
    }


def _summary_event(trace: RawTrace) -> dict[str, Any]:
    noncritical = [
        event
        for event in trace.events
        if event.event_id not in set(trace.required_evidence_ids)
    ]
    facts = [
        f"{event.source}.{event.operation} returned {json.dumps(event.payload, sort_keys=True)}"
        for event in noncritical
    ]
    return {
        "event_id": "handoff_summary",
        "step": 0,
        "source": "handoff_summarizer",
        "operation": "summarize",
        "payload": {
            "summary": " | ".join(facts) or "No independent detail was retained.",
            "source_event_ids": [event.event_id for event in noncritical],
        },
        "source_kind": "environment",
        "event_type": "handoff_summary",
        "integrity_hash": hashlib.sha256(
            json.dumps(facts, sort_keys=True).encode()
        ).hexdigest(),
    }


def _actionable_summary_event(trace: RawTrace) -> dict[str, Any]:
    """Keep provenance/context while projecting away answer-critical values."""
    required = set(trace.required_evidence_ids)
    items: list[dict[str, Any]] = []
    source_ids: list[str] = []
    for event in trace.events:
        if event.event_id == "subject_output":
            continue
        source_ids.append(event.event_id)
        if event.event_id in required:
            identifiers = {
                key: value
                for key, value in event.payload.items()
                if key.endswith("_id")
                or key in {
                    "id",
                    "order_id",
                    "transaction_id",
                    "sku",
                    "loan_id",
                    "package",
                }
            }
            items.append(
                {
                    "event_id": event.event_id,
                    "source": event.source,
                    "operation": event.operation,
                    "event_type": event.event_type,
                    "observed": True,
                    "identifiers": identifiers,
                    "available_fields": sorted(event.payload),
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
        "compression_policy": "structured_field_projection",
    }
    return {
        "event_id": "actionable_summary",
        "step": 0,
        "source": "handoff_summarizer",
        "operation": "summarize",
        "payload": payload,
        "source_kind": "environment",
        "event_type": "handoff_summary",
        "integrity_hash": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def build_view(trace: RawTrace, view_type: ViewType) -> tuple[MonitorView, PrivateGold]:
    required = set(trace.required_evidence_ids)
    events = list(trace.events)
    manifest: dict[str, Any] | None = None
    if view_type == "full":
        visible = [_event_dict(event) for event in events]
    elif view_type == "silent_filter":
        removed = next(iter(required))
        visible = [_event_dict(event) for event in events if event.event_id != removed]
    elif view_type == "explicit_redaction":
        removed = next(iter(required))
        visible = []
        for event in events:
            if event.event_id != removed:
                visible.append(_event_dict(event))
                continue
            redacted = _event_dict(event)
            redacted["payload"] = {"redacted": True}
            visible.append(redacted)
    elif view_type == "handoff_summary":
        visible = [_event_dict(event) for event in events if event.event_id == "subject_output"]
        visible.append(_summary_event(trace))
    elif view_type == "protected_summary":
        visible = [_event_dict(event) for event in events if event.event_id == "subject_output"]
        visible.append(_summary_event(trace))
        visible_types = {
            event["event_type"]
            for event in visible
            if event.get("event_type") != "subject_report"
        }
        manifest = {
            "expected_event_types": list(trace.expected_event_types),
            "present_event_types": sorted(visible_types),
            "missing_event_types": sorted(
                set(trace.expected_event_types) - visible_types
            ),
        }
    elif view_type == "actionable_summary":
        visible = [_event_dict(event) for event in events if event.event_id == "subject_output"]
        visible.append(_actionable_summary_event(trace))
    else:  # pragma: no cover - Literal keeps this unreachable for callers
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
        view_type=view_type,
        events=tuple(visible),
        completeness_manifest=manifest,
    )
    gold = PrivateGold(
        view_id=view_id,
        trace_id=trace.trace_id,
        dataset=trace.dataset,
        task_family=trace.task_family,
        gold_outcome=trace.gold_outcome,
        required_evidence_ids=trace.required_evidence_ids,
        expected_event_types=trace.expected_event_types,
        view_type=view_type,
    )
    return view, gold


def build_all_views(
    traces: tuple[RawTrace, ...],
    view_types: tuple[ViewType, ...] = (
        "full",
        "silent_filter",
        "explicit_redaction",
        "handoff_summary",
        "protected_summary",
    ),
) -> tuple[tuple[MonitorView, PrivateGold], ...]:
    return tuple(build_view(trace, view_type) for trace in traces for view_type in view_types)


def validate_views(
    traces: tuple[RawTrace, ...], pairs: tuple[tuple[MonitorView, PrivateGold], ...]
) -> list[str]:
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
        if forbidden & _recursive_keys(public):
            errors.append(f"{view.view_id}: private key leaked")
        if view.view_type == "full":
            visible_ids = [event["event_id"] for event in view.events]
            raw_ids = [event.event_id for event in raw.events]
            if visible_ids != raw_ids:
                errors.append(f"{view.view_id}: full view changed event order/content")
        if view.view_type == "silent_filter":
            visible_ids = {event["event_id"] for event in view.events}
            if not set(raw.required_evidence_ids) - visible_ids:
                errors.append(f"{view.view_id}: silent filter removed no required evidence")
        if view.view_type == "protected_summary":
            if not view.completeness_manifest:
                errors.append(f"{view.view_id}: protected manifest missing")
            elif not view.completeness_manifest.get("missing_event_types"):
                errors.append(f"{view.view_id}: protected manifest missed missing evidence")
        if view.view_type == "actionable_summary":
            summary_events = [
                event for event in view.events if event.get("event_id") == "actionable_summary"
            ]
            if len(summary_events) != 1:
                errors.append(f"{view.view_id}: actionable summary event missing or duplicated")
            else:
                summary_text = json.dumps(summary_events[0].get("payload", {}), sort_keys=True)
                if any(
                    value in summary_text
                    for value in ("gold_outcome", "mutation_id", "required_evidence_ids")
                ):
                    errors.append(f"{view.view_id}: actionable summary leaked private labels")
    view_types = {gold.view_type for _, gold in pairs}
    expected = len(traces) * len(view_types)
    if len(pairs) != expected:
        errors.append(f"expected {expected} views, got {len(pairs)}")
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
