from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MatrixThresholds:
    target_first_tool_min_delta: float = 0.03
    target_sequence_min_delta: float = 0.03
    open_first_tool_min_delta: float = -0.03
    open_sequence_min_delta: float = -0.03


@dataclass(frozen=True)
class AccuracySummary:
    first_tool_accuracy: float
    sequence_prefix_accuracy: float
    cases_total: int


def summarize_eval_case_score(payload: dict[str, Any]) -> AccuracySummary:
    policies = list(payload.get("policies", []))
    if not policies:
        return AccuracySummary(
            first_tool_accuracy=0.0,
            sequence_prefix_accuracy=0.0,
            cases_total=int(payload.get("cases_total", 0) or 0),
        )

    weighted_first = 0.0
    weighted_seq = 0.0
    weight_total = 0

    for row in policies:
        weight = int(row.get("cases_evaluated", 0) or 0)
        if weight <= 0:
            continue
        weighted_first += float(row.get("first_tool_accuracy", 0.0)) * weight
        weighted_seq += float(row.get("sequence_prefix_accuracy", 0.0)) * weight
        weight_total += weight

    if weight_total == 0:
        return AccuracySummary(
            first_tool_accuracy=0.0,
            sequence_prefix_accuracy=0.0,
            cases_total=int(payload.get("cases_total", 0) or 0),
        )

    return AccuracySummary(
        first_tool_accuracy=weighted_first / weight_total,
        sequence_prefix_accuracy=weighted_seq / weight_total,
        cases_total=weight_total,
    )


def gate_eval_case_delta(
    *,
    split: str,
    first_tool_delta: float,
    sequence_delta: float,
    thresholds: MatrixThresholds,
) -> str:
    split_norm = str(split).strip().lower()
    if split_norm not in {"target", "open"}:
        raise ValueError("split must be 'target' or 'open'")

    if split_norm == "target":
        pass_first = first_tool_delta >= float(thresholds.target_first_tool_min_delta)
        pass_seq = sequence_delta >= float(thresholds.target_sequence_min_delta)
    else:
        pass_first = first_tool_delta >= float(thresholds.open_first_tool_min_delta)
        pass_seq = sequence_delta >= float(thresholds.open_sequence_min_delta)

    checks = int(pass_first) + int(pass_seq)
    if checks == 2:
        return "PASS"
    if checks == 1:
        return "HOLD"
    return "FAIL"


def render_transfer_matrix_markdown(
    *,
    target_toolset_id: str,
    rows: list[dict[str, Any]],
    thresholds: MatrixThresholds,
    asset_paths: dict[str, str] | None = None,
) -> str:
    portfolio_verdict = "PASS"
    if any(str(row.get("verdict", "FAIL")).upper() == "FAIL" for row in rows):
        portfolio_verdict = "FAIL"
    elif any(str(row.get("verdict", "FAIL")).upper() == "HOLD" for row in rows):
        portfolio_verdict = "HOLD"

    worst_row = None
    if rows:
        worst_row = min(
            rows,
            key=lambda row: min(
                float(row.get("delta_first_tool_accuracy", 0.0)),
                float(row.get("delta_sequence_prefix_accuracy", 0.0)),
            ),
        )

    lines = [
        "## Toolset Transfer Matrix",
        "",
        f"Target toolset: {target_toolset_id}",
        "",
        "### Verdict Snapshot",
        "",
        f"- portfolio_verdict: {portfolio_verdict}",
        f"- rows_total: {len(rows)}",
    ]

    if isinstance(worst_row, dict):
        lines.append(
            f"- worst_toolset: {worst_row.get('toolset_id', 'unknown')} (first={float(worst_row.get('delta_first_tool_accuracy', 0.0)):+.4f}, sequence={float(worst_row.get('delta_sequence_prefix_accuracy', 0.0)):+.4f})"
        )

    if asset_paths:
        lines.extend(["", "### Assets", ""])
        for label, path in asset_paths.items():
            lines.append(f"- {label}: `{path}`")
        for path in asset_paths.values():
            if str(path).lower().endswith(".png"):
                lines.extend(["", f"![Transfer matrix asset]({path})"])

    lines.extend(
        [
        "",
        "| toolset | split | base_first | comparison_first | delta_first | base_seq | comparison_seq | delta_seq | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for row in rows:
        lines.append(
            "| "
            f"{row.get('toolset_id', 'unknown')} | "
            f"{row.get('split', 'open')} | "
            f"{float(row.get('base_first_tool_accuracy', 0.0)):.4f} | "
            f"{float(row.get('comparison_first_tool_accuracy', 0.0)):.4f} | "
            f"{float(row.get('delta_first_tool_accuracy', 0.0)):+.4f} | "
            f"{float(row.get('base_sequence_prefix_accuracy', 0.0)):.4f} | "
            f"{float(row.get('comparison_sequence_prefix_accuracy', 0.0)):.4f} | "
            f"{float(row.get('delta_sequence_prefix_accuracy', 0.0)):+.4f} | "
            f"{row.get('verdict', 'FAIL')} |"
        )

    lines.extend(
        [
            "",
            "### Gate Thresholds",
            "",
            f"- target first-tool delta >= {thresholds.target_first_tool_min_delta:+.3f}",
            f"- target sequence delta >= {thresholds.target_sequence_min_delta:+.3f}",
            f"- open first-tool delta >= {thresholds.open_first_tool_min_delta:+.3f}",
            f"- open sequence delta >= {thresholds.open_sequence_min_delta:+.3f}",
            "",
            "Verdict rule: PASS if both checks pass, HOLD if exactly one passes, FAIL otherwise.",
            "",
        ]
    )

    return "\n".join(lines)
