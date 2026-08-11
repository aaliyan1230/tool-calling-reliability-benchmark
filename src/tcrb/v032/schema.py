from __future__ import annotations

from typing import Literal

# The dataclasses are reused verbatim from v3.1. Only the view vocabulary and the
# factor labelling are new here, so redefining the containers would just create
# two shapes that have to be kept in sync.
from tcrb.v031.schema import (  # noqa: F401  (re-exported for v032 callers)
    GOLD_OUTCOMES,
    GoldOutcome,
    MonitorResult,
    MonitorView,
    PrivateGold,
    RawTrace,
    TraceEvent,
    Verdict,
)


ViewType = Literal[
    "full",
    "lossless_summary",
    "raw_projection",
    "actionable_summary",
]

VIEW_TYPES: tuple[ViewType, ...] = (
    "full",
    "lossless_summary",
    "raw_projection",
    "actionable_summary",
)

# The 2x2. `packaging` is how the trace is dressed; `content` is whether the
# answer-critical values survive. Analysis reads main effects off these labels
# rather than hard-coding view names.
PACKAGING: dict[str, str] = {
    "full": "raw",
    "raw_projection": "raw",
    "lossless_summary": "summary",
    "actionable_summary": "summary",
}

CONTENT: dict[str, str] = {
    "full": "values_present",
    "lossless_summary": "values_present",
    "raw_projection": "values_stripped",
    "actionable_summary": "values_stripped",
}


def factors(view_type: str) -> dict[str, str]:
    return {"packaging": PACKAGING[view_type], "content": CONTENT[view_type]}
