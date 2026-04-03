from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from .finetune.evaluate import compare_run_payloads


CORE_METRICS = ("task_success_rate", "invalid_tool_call_rate")


@dataclass(frozen=True)
class StudyGateThresholds:
    flatline_epsilon: float = 1e-4
    min_effect_vs_null: float = 3e-3
    matrix_flatline_epsilon: float = 1e-4
    require_matrix_signal: bool = False
    require_matrix_not_fail: bool = False


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _max_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(abs(value) for value in values))


def _metric_deltas(
    comparison_payload: dict[str, Any], metric_name: str
) -> list[float]:
    values: list[float] = []
    for row in comparison_payload.get("policies", []):
        delta = row.get("delta", {})
        value = delta.get(metric_name)
        if value is None:
            continue
        values.append(float(value))
    return values


def summarize_delta_signal(comparison_payload: dict[str, Any]) -> dict[str, Any]:
    success_deltas = _metric_deltas(comparison_payload, "task_success_rate")
    invalid_deltas = _metric_deltas(comparison_payload, "invalid_tool_call_rate")
    all_deltas = success_deltas + invalid_deltas

    return {
        "policy_rows": int(len(comparison_payload.get("policies", []))),
        "mean_success_delta": _mean(success_deltas),
        "mean_invalid_delta": _mean(invalid_deltas),
        "mean_abs_success_delta": _mean([abs(value) for value in success_deltas]),
        "mean_abs_invalid_delta": _mean([abs(value) for value in invalid_deltas]),
        "max_abs_core_delta": _max_abs(all_deltas),
    }


def summarize_advantage_vs_null(
    ft_vs_base_payload: dict[str, Any],
    null_vs_base_payload: dict[str, Any],
) -> dict[str, Any]:
    ft_map = {
        str(row.get("policy", "")): dict(row.get("delta", {}))
        for row in ft_vs_base_payload.get("policies", [])
    }
    null_map = {
        str(row.get("policy", "")): dict(row.get("delta", {}))
        for row in null_vs_base_payload.get("policies", [])
    }
    policies = sorted(set(ft_map.keys()) & set(null_map.keys()))

    advantage_values: list[float] = []
    by_metric: dict[str, list[float]] = {metric: [] for metric in CORE_METRICS}
    for policy in policies:
        ft_delta = ft_map.get(policy, {})
        null_delta = null_map.get(policy, {})
        for metric in CORE_METRICS:
            ft_value = ft_delta.get(metric)
            null_value = null_delta.get(metric)
            if ft_value is None or null_value is None:
                continue
            advantage = float(ft_value) - float(null_value)
            by_metric[metric].append(advantage)
            advantage_values.append(advantage)

    return {
        "policies_compared": int(len(policies)),
        "mean_advantage_success_delta": _mean(by_metric["task_success_rate"]),
        "mean_advantage_invalid_delta": _mean(by_metric["invalid_tool_call_rate"]),
        "max_abs_advantage_delta": _max_abs(advantage_values),
    }


def summarize_matrix_signal(matrix_payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(matrix_payload.get("rows", []))
    first_deltas: list[float] = []
    seq_deltas: list[float] = []
    verdict_counts = {"PASS": 0, "HOLD": 0, "FAIL": 0}

    for row in rows:
        first_delta = row.get("delta_first_tool_accuracy")
        seq_delta = row.get("delta_sequence_prefix_accuracy")
        if first_delta is not None:
            first_deltas.append(float(first_delta))
        if seq_delta is not None:
            seq_deltas.append(float(seq_delta))
        verdict = str(row.get("verdict", "")).upper()
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

    return {
        "rows_total": int(len(rows)),
        "portfolio_verdict": str(matrix_payload.get("portfolio_verdict", "")),
        "max_abs_delta": _max_abs(first_deltas + seq_deltas),
        "mean_abs_first_delta": _mean([abs(value) for value in first_deltas]),
        "mean_abs_sequence_delta": _mean([abs(value) for value in seq_deltas]),
        "verdict_counts": verdict_counts,
    }


def evaluate_study_gates(
    *,
    base_payload: dict[str, Any],
    finetuned_payload: dict[str, Any],
    thresholds: StudyGateThresholds,
    null_payload: dict[str, Any] | None = None,
    matrix_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ft_vs_base = compare_run_payloads(base_payload, finetuned_payload)
    ft_summary = summarize_delta_signal(ft_vs_base)

    checks: list[dict[str, Any]] = []

    base_vs_ft_nonflat = bool(ft_summary["max_abs_core_delta"] > thresholds.flatline_epsilon)
    checks.append(
        {
            "name": "base_vs_ft_nonflatline",
            "passed": base_vs_ft_nonflat,
            "value": float(ft_summary["max_abs_core_delta"]),
            "threshold": float(thresholds.flatline_epsilon),
            "detail": "max abs core delta must exceed flatline epsilon",
        }
    )

    null_summary: dict[str, Any] | None = None
    if null_payload is not None:
        null_vs_base = compare_run_payloads(base_payload, null_payload)
        null_summary = summarize_advantage_vs_null(ft_vs_base, null_vs_base)
        effect_vs_null = bool(
            null_summary["max_abs_advantage_delta"] >= thresholds.min_effect_vs_null
        )
        checks.append(
            {
                "name": "ft_distinct_from_null_control",
                "passed": effect_vs_null,
                "value": float(null_summary["max_abs_advantage_delta"]),
                "threshold": float(thresholds.min_effect_vs_null),
                "detail": "max abs (ft-base) - (null-base) must exceed min effect",
            }
        )

    matrix_summary: dict[str, Any] | None = None
    if matrix_payload is not None:
        matrix_summary = summarize_matrix_signal(matrix_payload)
        if thresholds.require_matrix_signal:
            matrix_has_signal = bool(
                matrix_summary["max_abs_delta"] > thresholds.matrix_flatline_epsilon
            )
            checks.append(
                {
                    "name": "matrix_nonflatline",
                    "passed": matrix_has_signal,
                    "value": float(matrix_summary["max_abs_delta"]),
                    "threshold": float(thresholds.matrix_flatline_epsilon),
                    "detail": "max abs transfer-matrix delta must exceed epsilon",
                }
            )
        if thresholds.require_matrix_not_fail:
            matrix_not_fail = str(matrix_summary.get("portfolio_verdict", "")).upper() != "FAIL"
            checks.append(
                {
                    "name": "matrix_portfolio_not_fail",
                    "passed": matrix_not_fail,
                    "value": str(matrix_summary.get("portfolio_verdict", "")),
                    "threshold": "not FAIL",
                    "detail": "transfer-matrix portfolio verdict must not be FAIL",
                }
            )
    elif thresholds.require_matrix_signal or thresholds.require_matrix_not_fail:
        checks.append(
            {
                "name": "matrix_input_required",
                "passed": False,
                "value": "missing",
                "threshold": "provided",
                "detail": "matrix JSON required by selected thresholds",
            }
        )

    verdict = "PASS" if all(bool(check["passed"]) for check in checks) else "FAIL"
    return {
        "type": "study_gate",
        "verdict": verdict,
        "thresholds": {
            "flatline_epsilon": float(thresholds.flatline_epsilon),
            "min_effect_vs_null": float(thresholds.min_effect_vs_null),
            "matrix_flatline_epsilon": float(thresholds.matrix_flatline_epsilon),
            "require_matrix_signal": bool(thresholds.require_matrix_signal),
            "require_matrix_not_fail": bool(thresholds.require_matrix_not_fail),
        },
        "base_vs_finetuned": ft_summary,
        "null_control": null_summary,
        "matrix": matrix_summary,
        "checks": checks,
    }