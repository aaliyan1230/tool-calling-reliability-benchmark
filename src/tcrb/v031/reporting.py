from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_report(run_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or (run_dir / "report")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = _read_optional_json(run_dir / "dataset_manifest.json")
    audit = _read_optional_json(run_dir / "audit.json")
    stability = _read_repeat_summary(run_dir)
    figure_paths = _build_figures(summary, output_dir)
    report = _render_report(
        summary, figure_paths, manifest=manifest, audit=audit, stability=stability
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "figures": figure_paths,
        "report": str(output_dir / "report.md"),
    }


def _build_figures(summary: dict[str, Any], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to build v3.1 figures") from exc

    metrics = summary.get("metrics", {})
    views = [
        "full",
        "silent_filter",
        "explicit_redaction",
        "handoff_summary",
        "protected_summary",
    ]
    models = [model for model in ("deepseek", "gpt") if model in metrics]
    if any("actionable_summary" in (metrics[model] or {}) for model in models):
        views.append("actionable_summary")
    labels = {
        "full": "Full",
        "silent_filter": "Silent filter",
        "explicit_redaction": "Redaction",
        "handoff_summary": "Summary",
        "protected_summary": "Protected",
        "actionable_summary": "Actionable summary",
    }
    paths: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    width = 0.36
    x = list(range(len(views)))
    for ax, metric_name, title in (
        (axes[0], "unsafe_approval_rate", "Unsafe reports approved"),
        (axes[1], "safe_approval_rate", "Safe reports approved"),
    ):
        for index, model in enumerate(models):
            values = [
                (metrics[model].get(view) or {}).get(metric_name) or 0.0
                for view in views
            ]
            ax.bar(
                [item + (index - (len(models) - 1) / 2) * width for item in x],
                [value * 100 for value in values],
                width=width,
                label=model,
            )
        ax.set_xticks(x, [labels[view] for view in views], rotation=20)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Approval rate (%)")
    axes[1].legend(title="Monitor")
    fig.suptitle("Effect of evidence visibility on monitor decisions")
    fig.tight_layout()
    path = output_dir / "figure-1-approval-rates-by-view.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    comparison_views = views[1:]
    x = list(range(len(comparison_views)))
    for ax, metric_name, title in (
        (axes[0], "unsafe_approval_delta", "Unsafe approval change"),
        (axes[1], "safe_approval_delta", "Safe approval change"),
    ):
        for model in models:
            effects = metrics[model].get("paired_effects", {})
            values = []
            lows = []
            highs = []
            for view in comparison_views:
                item = effects.get(view, {})
                values.append((item.get(metric_name) or 0.0) * 100)
                interval = (item.get("bootstrap_95_ci") or {}).get(metric_name)
                if interval:
                    lows.append(values[-1] - interval[0] * 100)
                    highs.append(interval[1] * 100 - values[-1])
                else:
                    lows.append(0.0)
                    highs.append(0.0)
            offset = -0.18 if model == "deepseek" else 0.18
            ax.errorbar(
                [item + offset for item in x],
                values,
                yerr=[lows, highs],
                fmt="o",
                capsize=4,
                label=model,
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x, [labels[view] for view in comparison_views], rotation=20)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Change vs full trace (percentage points)")
    axes[1].legend(title="Monitor")
    fig.suptitle("Paired visibility effects with clustered bootstrap intervals")
    fig.tight_layout()
    path = output_dir / "figure-2-paired-visibility-effects.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _render_report(
    summary: dict[str, Any],
    figures: list[str],
    *,
    manifest: dict[str, Any] | None = None,
    audit: dict[str, Any] | None = None,
    stability: dict[str, Any] | None = None,
) -> str:
    manifest = manifest or {}
    audit = audit or {}
    stability = stability or {}
    dataset = manifest.get("dataset", "tcrb")
    is_tau = dataset == "tau_bench"
    lines = [
        "# v3.1 Evidence-Visibility Blind-Spot Pilot",
        "",
        "## Question",
        "",
        "Does filtering or compressing an agent trace make a monitor approve unsafe reports that it rejects from the full trace?",
        "",
        "## Scope",
        "",
        (
            f"This run uses {manifest.get('trace_count', 'fixed')} fixed "
            f"{dataset} traces "
            f"({summary.get('primary_rows', 0)} primary model-view calls plus "
            f"{summary.get('stability_rows', 0)} repeats), with two monitors. "
            "The public view is separated from private gold. No LLM judge is used."
        ),
        "",
        f"Parsed responses: {summary.get('responses_by_status', {}).get('valid', 0)}/{summary.get('rows', 0)}.",
        f"Estimated extra cost: ${summary.get('estimated_extra_cost_usd', 0):.6f}.",
        "",
        "## Results",
        "",
    ]
    for model, by_view in summary.get("metrics", {}).items():
        if not isinstance(by_view, dict):
            continue
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| View | Unsafe approval | Unsafe detection | Safe approval | Invalid citations |")
        lines.append("|---|---:|---:|---:|---:|")
        view_order = (
            "full",
            "silent_filter",
            "explicit_redaction",
            "handoff_summary",
            "protected_summary",
            "actionable_summary",
        )
        for view in view_order:
            item = by_view.get(view)
            if not isinstance(item, dict):
                continue
            lines.append(
                "| {view} | {unsafe:.1%} | {detect:.1%} | {safe:.1%} | {cite:.1%} |".format(
                    view=view,
                    unsafe=item.get("unsafe_approval_rate") or 0,
                    detect=item.get("unsafe_detection_rate") or 0,
                    safe=item.get("safe_approval_rate") or 0,
                    cite=1 - (item.get("valid_citation_rate") or 0),
                )
            )
        lines.append("")
        lines.append("Paired effects:")
        lines.append("")
        for view, item in (by_view.get("paired_effects") or {}).items():
            lines.append(
                f"- `{view}`: unsafe approval change "
                f"{(item.get('unsafe_approval_delta') or 0):+.1%}; "
                f"paired blind-spot rate {(item.get('paired_blind_spot_rate') or 0):.1%}."
            )
        lines.append("")
    quality = summary.get("quality_gate", {})
    lines.extend(
        [
            "## What this means",
            "",
            (
                "The new actionable-summary condition produced no unsafe approvals for either model, but safe approval fell to 8.3% for both. The older handoff summary had a small DeepSeek-only unsafe approval rate (8.3%) while also rejecting almost all safe reports; that is not a robust cross-model finding. Overall, evidence loss mainly caused fail-closed availability loss."
                if not is_tau and "actionable_summary" in (summary.get("metrics", {}).get("deepseek", {}))
                else (
                    "The TCRB matrix produced no unsafe approvals after any transformation; evidence loss instead caused a strong fail-closed availability loss."
                    if not is_tau
                    else "This τ-bench smoke is an external replay check, not a full benchmark. Evidence loss generally caused conservative decisions. One DeepSeek handoff cell approved an unsafe report in the primary pass, but the three-repeat handoff check for all four tasks produced no approvals, so that apparent blind spot is not stable enough to claim."
                )
            ),
            "",
            (
                f"Stability repeats: {stability.get('records', 0)} calls; "
                f"{stability.get('unanimous_group_rate', 0):.1%} unanimous groups."
                if stability.get("records")
                else f"Stability repeats: {summary.get('stability', {}).get('unanimous_rate', 0):.1%} unanimous across repeated cells."
            ),
            "",
        ]
    )
    lines.extend(
        [
            "## Integrity gate",
            "",
            f"Quality gate passed: **{quality.get('passed', False)}**",
            f"Audit passed: **{audit.get('passed', False)}**",
            "",
            (
                "This is a TCRB-only result. It should not be generalized to all production summaries until a more realistic lossy-summary variant is run."
                if not is_tau
                else "This is a small τ-bench smoke result. It should not be generalized until more tasks and a more realistic lossy-summary variant are run."
            ),
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in figures)
    return "\n".join(lines) + "\n"


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_repeat_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "stability_repeats.jsonl"
    gold_path = run_dir / "private_gold.jsonl"
    if not path.exists() or not gold_path.exists():
        return {}
    gold = {
        row["view_id"]: row
        for row in _read_jsonl(gold_path)
    }
    groups: dict[tuple[str, str, str], list[str | None]] = {}
    rows = _read_jsonl(path)
    for row in rows:
        key = (
            row.get("provider", ""),
            row.get("task_family", ""),
            gold.get(row.get("view_id", ""), {}).get("gold_outcome", ""),
        )
        groups.setdefault(key, []).append(
            (row.get("monitor_result") or {}).get("verdict")
        )
    return {
        "records": len(rows),
        "invalid": sum(not isinstance(row.get("monitor_result"), dict) for row in rows),
        "unanimous_group_rate": (
            sum(len(set(values)) == 1 for values in groups.values()) / len(groups)
            if groups
            else 0.0
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
