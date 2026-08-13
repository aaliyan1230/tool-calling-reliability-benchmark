from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


GoldOutcome = Literal["safe", "unsafe"]
ViewType = Literal[
    "full",
    "silent_filter",
    "explicit_redaction",
    "handoff_summary",
    "protected_summary",
    "actionable_summary",
]
Verdict = Literal["approve", "reject", "insufficient_evidence"]

GOLD_OUTCOMES = ("safe", "unsafe")
VIEW_TYPES = (
    "full",
    "silent_filter",
    "explicit_redaction",
    "handoff_summary",
    "protected_summary",
)

FOLLOWUP_VIEW_TYPES = VIEW_TYPES + ("actionable_summary",)


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    step: int
    source: str
    operation: str
    payload: dict[str, Any]
    source_kind: Literal["subject", "independent", "environment"]
    event_type: str
    integrity_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawTrace:
    trace_id: str
    dataset: Literal["tcrb", "tau_bench"]
    domain: str
    task_id: str
    task_family: str
    claim: str
    events: tuple[TraceEvent, ...]
    gold_outcome: GoldOutcome
    required_evidence_ids: tuple[str, ...]
    expected_event_types: tuple[str, ...]
    mutation_id: str
    source_commit: str

    def to_private_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events"] = [event.to_dict() for event in self.events]
        payload["required_evidence_ids"] = list(self.required_evidence_ids)
        payload["expected_event_types"] = list(self.expected_event_types)
        return payload


@dataclass(frozen=True)
class MonitorView:
    view_id: str
    trace_id: str
    dataset: str
    domain: str
    task_id: str
    task_family: str
    claim: str
    view_type: ViewType
    events: tuple[dict[str, Any], ...]
    completeness_manifest: dict[str, Any] | None = None

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "view_id": self.view_id,
            "trace_id": self.trace_id,
            "dataset": self.dataset,
            "domain": self.domain,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "claim": self.claim,
            "events": list(self.events),
        }
        if self.completeness_manifest is not None:
            payload["completeness_manifest"] = self.completeness_manifest
        return payload


@dataclass(frozen=True)
class PrivateGold:
    view_id: str
    trace_id: str
    dataset: str
    task_family: str
    gold_outcome: GoldOutcome
    required_evidence_ids: tuple[str, ...]
    expected_event_types: tuple[str, ...]
    view_type: ViewType

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_evidence_ids"] = list(self.required_evidence_ids)
        payload["expected_event_types"] = list(self.expected_event_types)
        return payload


@dataclass(frozen=True)
class MonitorResult:
    verdict: Verdict
    evidence_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ScoreRow:
    call_id: str
    model: str
    view_id: str
    trace_id: str
    dataset: str
    domain: str
    task_family: str
    gold_outcome: GoldOutcome
    view_type: ViewType
    verdict: Verdict | None
    valid: bool
    unsafe_approved: bool | None
    safe_approved: bool | None
    safe_correct: bool | None
    cited_ids_valid: bool | None
    required_evidence_used: bool | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
