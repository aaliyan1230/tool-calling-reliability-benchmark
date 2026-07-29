from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

# ── Actions ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: _new_call_id())


@dataclass(frozen=True)
class FinalAnswer:
    text: str


@dataclass(frozen=True)
class Clarify:
    text: str


@dataclass(frozen=True)
class Abort:
    reason: str


AgentAction = ToolCall | FinalAnswer | Clarify | Abort

# ── Observations ─────────────────────────────────────────────────────

ObservationStatus = Literal[
    "success",
    "execution_error",
    "timeout",
    "rate_limit",
    "schema_drift",
    "partial_output",
    "silent_corruption",
    "cross_source_conflict",
    "invalid_arguments",
    "unknown_tool",
    "budget_exhausted",
]


@dataclass(frozen=True)
class Observation:
    status: ObservationStatus
    payload: Any | None = None
    latency_ms: int = 0
    cost_usd: float = 0.0
    call_id: str = ""

    @property
    def is_error(self) -> bool:
        return self.status != "success"


# ── Trace records ────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepRecord:
    step_index: int
    raw_model_output: str
    parsed_action: AgentAction | None
    parse_error: str | None
    observation: Observation | None
    tokens_in: int = 0
    tokens_out: int = 0
    timing_ms: int = 0
    cost_usd: float = 0.0


DiagnosticLabel = Literal[
    "tool_skip",
    "unnecessary_tool_use",
    "result_ignore",
    "fabrication",
    "retry_loop",
    "wrong_fallback",
    "silent_result_trust",
    "premature_stop",
    "argument_invalid",
    "unexecutable_call",
]


@dataclass
class EpisodeTrace:
    task_id: str
    domain: str
    steps: list[StepRecord] = field(default_factory=list)
    final_response: str | None = None
    success: bool = False
    diagnostic_labels: list[DiagnosticLabel] = field(default_factory=list)
    faults_applied: list[str] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_time_ms: int = 0


# ── Tool definitions ─────────────────────────────────────────────────

ToolExecutor = Callable[..., Any]


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    base_latency_ms: int = 50
    jitter_ms: int = 10
    timeout_ms: int = 2000
    executor: ToolExecutor | None = None
    fault_multipliers: dict[str, float] = field(default_factory=dict)

    def to_openai_function(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


# ── Task definitions ─────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskDef:
    task_id: str
    domain: str
    user_query: str
    category: Literal["tool_required", "no_tool", "missing_information"]
    available_tools: list[str]
    canonical_claims: list[str] = field(default_factory=list)
    valid_tool_sequences: list[list[str]] = field(default_factory=list)
    fault_variants: list[dict[str, Any]] = field(default_factory=list)


# ── Fault schedules ──────────────────────────────────────────────────


@dataclass(frozen=True)
class FaultSchedule:
    task_id: str
    fault_type: str
    step_index: int
    tool_name: str
    config: dict[str, Any]


# ── Episode result ───────────────────────────────────────────────────


@dataclass(frozen=True)
class EpisodeResult:
    task_id: str
    domain: str
    success: bool
    fault_type: str | None
    diagnostic_labels: list[DiagnosticLabel]
    steps: int
    total_tokens: int
    total_cost_usd: float
    total_time_ms: int
    calls_made: int
    recovery_steps: int


# ── Helpers ──────────────────────────────────────────────────────────

_call_counter: int = 0


def _new_call_id() -> str:
    global _call_counter
    _call_counter += 1
    return f"call_{_call_counter}"


def reset_call_counter() -> None:
    global _call_counter
    _call_counter = 0
