from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


Domain = Literal[
    "customer_support",
    "ecommerce",
    "fintech",
    "developer_tools",
]
PayloadState = Literal["correct", "corrupt"]
SelfLabel = Literal["verified", "warning"]
PromptVariant = Literal["baseline", "evidence_first"]
Decision = Literal["allow", "review", "block"]
SourceKind = Literal["subject_under_review", "independent"]


@dataclass(frozen=True)
class EvidenceEvent:
    evidence_id: str
    source: str
    source_kind: SourceKind
    payload: dict[str, Any]
    self_assessment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.self_assessment is None:
            payload.pop("self_assessment")
        return payload


@dataclass(frozen=True)
class VerifierView:
    case_id: str
    domain: Domain
    user_goal: str
    events: tuple[EvidenceEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "user_goal": self.user_goal,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(frozen=True)
class PrivateGold:
    case_id: str
    base_case_id: str
    domain: Domain
    payload_state: PayloadState
    self_label: SelfLabel
    expected_flag: bool
    answer_critical_field: str
    correct_value: Any
    visible_value: Any
    relevant_independent_evidence_ids: tuple[str, ...]
    corruption_reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["relevant_independent_evidence_ids"] = list(
            self.relevant_independent_evidence_ids
        )
        return payload


@dataclass(frozen=True)
class PilotVariant:
    verifier_view: VerifierView
    private_gold: PrivateGold


@dataclass(frozen=True)
class MonitorResult:
    decision: Decision
    confidence: float
    evidence_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
        }


MONITOR_RESULT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["allow", "review", "block"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        },
        "reason": {"type": "string", "maxLength": 400},
    },
    "required": ["decision", "confidence", "evidence_ids", "reason"],
}
