"""Pure metric helpers for the v0.2 reliability study.

These helpers stay free of torch/transformers imports so the aggregation
rules behind the study's numbers can be unit-tested in the dev-only
environment used by CI.
"""

from __future__ import annotations

from typing import Any


def is_recovery(episode: dict[str, Any]) -> bool:
    """A faulted episode counts as recovery only if it succeeded after its
    scheduled fault was actually applied."""
    return bool(episode["success"]) and bool(episode["fault_applied"])


def count_diagnostic_labels(results: list[dict[str, Any]]) -> dict[str, int]:
    """Tally diagnostic labels across every clean and faulted episode."""
    counts: dict[str, int] = {}
    for record in results:
        for label in record["clean"]["diagnostic_labels"]:
            counts[label] = counts.get(label, 0) + 1
        for fault in record.get("faulted", []):
            for label in fault.get("diagnostic_labels", []):
                counts[label] = counts.get(label, 0) + 1
    return counts
