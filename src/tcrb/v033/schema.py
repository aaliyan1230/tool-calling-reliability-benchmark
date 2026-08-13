from __future__ import annotations

from typing import Literal

from tcrb.v031.schema import (  # noqa: F401  (re-exported for v033 callers)
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
    "symmetric_advertised",
    "symmetric_silent",
    "asymmetric_advertised",
    "asymmetric_silent",
]

VIEW_TYPES: tuple[ViewType, ...] = (
    "full",
    "symmetric_advertised",
    "symmetric_silent",
    "asymmetric_advertised",
    "asymmetric_silent",
)

# The four degraded cells. `full` sits outside the 2x2 as the baseline the
# quality gate is read off.
DEGRADED_VIEW_TYPES: tuple[ViewType, ...] = (
    "symmetric_advertised",
    "symmetric_silent",
    "asymmetric_advertised",
    "asymmetric_silent",
)

# How much is removed.
SCOPE: dict[str, str] = {
    "symmetric_advertised": "symmetric",
    "symmetric_silent": "symmetric",
    "asymmetric_advertised": "asymmetric",
    "asymmetric_silent": "asymmetric",
}

# Whether the view admits that something was removed.
ANNOUNCEMENT: dict[str, str] = {
    "symmetric_advertised": "advertised",
    "asymmetric_advertised": "advertised",
    "symmetric_silent": "silent",
    "asymmetric_silent": "silent",
}


def factors(view_type: str) -> dict[str, str | None]:
    return {"scope": SCOPE.get(view_type), "announcement": ANNOUNCEMENT.get(view_type)}
