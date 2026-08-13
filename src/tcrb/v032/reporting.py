from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analysis import CONTRASTS
from .schema import VIEW_TYPES


VIEW_LABELS = {
    "full": "Full\n(raw, values kept)",
    "lossless_summary": "Lossless summary\n(summary, values kept)",
    "raw_projection": "Raw projection\n(raw, values stripped)",
    "actionable_summary": "Actionable summary\n(summary, values stripped)",
}

SHORT_LABELS = {
    "full": "Full",
    "lossless_summary": "Lossless\nsummary",
    "raw_projection": "Raw\nprojection",
    "actionable_summary": "Actionable\nsummary",
}

MODEL_LABELS = {"deepseek": "DeepSeek V4 Flash", "gpt": "GPT-5.6 Terra"}

CONTRAST_LABELS = {
    "packaging_effect_values_present": "Packaging\n(values kept)",
    "packaging_effect_values_stripped": "Packaging\n(values stripped)",
    "content_effect_raw": "Content\n(raw packaging)",
    "content_effect_summary": "Content\n(summary packaging)",
}


def build_report(run_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or (run_dir / "report")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = _read_optional(run_dir / "dataset_manifest.json")
    audit = _read_optional(run_dir / "audit.json")
    figures = _build_figures(summary, output_dir)
    (output_dir / "report.md").write_text(
        _render(summary, figures, manifest=manifest, audit=audit), encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "figures": figures,
        "report": str(output_dir / "report.md"),
    }


def _build_figures(summary: dict[str, Any], output_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to build v3.2 figures") from exc

    metrics = summary.get("metrics", {})
    models = [m for m in ("deepseek", "gpt") if m in metrics]
    paths: list[str] = []

    # Figure 1 - the headline. Safe and unsafe approval for all four cells.
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.6), squeeze=False)
    for axis, model in zip(axes[0], models):
        by_view = metrics[model]["by_view"]
        views = [v for v in VIEW_TYPES if v in by_view]
        x = range(len(views))
        width = 0.38
        safe = [(by_view[v].get("safe_approval_rate") or 0) * 100 for v in views]
        unsafe = [(by_view[v].get("unsafe_approval_rate") or 0) * 100 for v in views]
        axis.bar([i - width / 2 for i in x], safe, width, label="Safe approved (want high)", color="#2f7ed8")
        axis.bar([i + width / 2 for i in x], unsafe, width, label="Unsafe approved (want zero)", color="#c8102e")
        for i, value in enumerate(safe):
            axis.text(i - width / 2, value + 1.5, f"{value:.0f}%", ha="center", fontsize=8)
        for i, value in enumerate(unsafe):
            axis.text(i + width / 2, value + 1.5, f"{value:.0f}%", ha="center", fontsize=8)
        axis.set_xticks(list(x))
        axis.set_xticklabels([SHORT_LABELS[v] for v in views], fontsize=8)
        axis.set_ylim(0, 112)
        axis.set_ylabel("Rate (%)")
        axis.set_title(MODEL_LABELS.get(model, model))
        axis.legend(fontsize=8, loc="upper right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Does the monitor react to missing values, or to summary packaging?", fontsize=12
    )
    fig.tight_layout()
    path = output_dir / "figure-1-two-by-two-approval.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(str(path))

    # Figure 2 - the four one-factor contrasts, with clustered bootstrap CIs.
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.4), squeeze=False)
    names = [name for name, _, _, _ in CONTRASTS]
    for axis, model in zip(axes[0], models):
        contrasts = metrics[model]["contrasts"]
        values, lows, highs, colors = [], [], [], []
        for name in names:
            entry = contrasts.get(name, {})
            delta = (entry.get("safe_approval_delta") or 0) * 100
            ci = (entry.get("bootstrap_95_ci") or {}).get("safe_approval_delta")
            low, high = (ci[0] * 100, ci[1] * 100) if ci else (delta, delta)
            values.append(delta)
            lows.append(max(0.0, delta - low))
            highs.append(max(0.0, high - delta))
            colors.append("#e07b39" if entry.get("factor_moved") == "packaging" else "#3d7d3f")
        axis.bar(range(len(names)), values, yerr=[lows, highs], capsize=4, color=colors)
        axis.axhline(0, color="black", linewidth=0.9)
        for i, value in enumerate(values):
            # Clear the whisker: sit above the upper cap for negative bars,
            # above the bar itself for zero/positive ones.
            top = value + highs[i]
            axis.text(
                i,
                (top + 3) if value < 0 else (value + 3),
                f"{value:+.0f}pp",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels([CONTRAST_LABELS[n] for n in names], fontsize=8)
        axis.set_ylabel("Change in safe-approval rate (pp)")
        axis.set_ylim(-105, 30)
        axis.set_title(MODEL_LABELS.get(model, model))
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Moving one factor at a time (orange = packaging, green = content)", fontsize=12
    )
    fig.tight_layout()
    path = output_dir / "figure-2-one-factor-contrasts.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(str(path))

    # Figure 3 - each factor averaged over both levels of the other.
    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 4.2), squeeze=False)
    for axis, model in zip(axes[0], models):
        main = metrics[model]["main_effects"]
        groups = [
            ("Raw\npackaging", main["packaging"].get("raw", {})),
            ("Summary\npackaging", main["packaging"].get("summary", {})),
            ("Values\nkept", main["content"].get("values_present", {})),
            ("Values\nstripped", main["content"].get("values_stripped", {})),
        ]
        values = [(entry.get("safe_approval_rate") or 0) * 100 for _, entry in groups]
        colors = ["#e07b39", "#e07b39", "#3d7d3f", "#3d7d3f"]
        axis.bar(range(len(groups)), values, color=colors)
        for i, value in enumerate(values):
            axis.text(i, value + 1.5, f"{value:.0f}%", ha="center", fontsize=9)
        axis.set_xticks(range(len(groups)))
        axis.set_xticklabels([label for label, _ in groups], fontsize=8)
        axis.set_ylim(0, 112)
        axis.set_ylabel("Safe-approval rate (%)")
        axis.set_title(MODEL_LABELS.get(model, model))
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Main effect of each factor (orange = packaging, green = content)", fontsize=12)
    fig.tight_layout()
    path = output_dir / "figure-3-main-effects.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(str(path))

    return paths


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _pp(value: Any) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}pp"


def _ci(entry: dict[str, Any], key: str) -> str:
    ci = (entry.get("bootstrap_95_ci") or {}).get(key)
    return "n/a" if not ci else f"[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]"


def _render(
    summary: dict[str, Any],
    figures: list[str],
    *,
    manifest: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    metrics = summary.get("metrics", {})
    models = [m for m in ("deepseek", "gpt") if m in metrics]
    lines: list[str] = [
        "# v3.2 — is fail-closed about missing values or about summary packaging?",
        "",
        f"Responses: {summary.get('rows')} "
        f"({summary.get('primary_rows')} primary, {summary.get('stability_rows')} repeats). "
        f"Valid: {summary.get('responses_by_status', {}).get('valid')}. "
        f"Cost: ${summary.get('estimated_extra_cost_usd', 0):.4f}.",
        "",
        f"Quality gate: **{'PASS' if summary.get('quality_gate', {}).get('passed') else 'FAIL'}**"
        + (
            f" · Integrity audit: **{'PASS' if audit.get('passed') else 'FAIL'}**"
            if audit
            else ""
        ),
        "",
        "## The design",
        "",
        "|  | Values kept | Values stripped |",
        "|---|---|---|",
        "| **Raw packaging** | `full` | `raw_projection` |",
        "| **Summary packaging** | `lossless_summary` | `actionable_summary` |",
        "",
        "`full` and `actionable_summary` are carried over unchanged from v3.1, so their",
        "numbers double as a replication check.",
        "",
        "## Rates by cell",
        "",
    ]
    for model in models:
        by_view = metrics[model]["by_view"]
        lines += [
            f"### {MODEL_LABELS.get(model, model)}",
            "",
            "| View | Packaging | Values | Safe approved | Unsafe approved | Insufficient evidence |",
            "|---|---|---|---:|---:|---:|",
        ]
        for view in VIEW_TYPES:
            entry = by_view.get(view)
            if not entry:
                continue
            from .schema import CONTENT, PACKAGING

            lines.append(
                f"| `{view}` | {PACKAGING[view]} | "
                f"{'kept' if CONTENT[view] == 'values_present' else 'stripped'} | "
                f"{_pct(entry.get('safe_approval_rate'))} | "
                f"{_pct(entry.get('unsafe_approval_rate'))} | "
                f"{_pct(entry.get('insufficient_evidence_rate'))} |"
            )
        lines.append("")

    lines += ["## Moving one factor at a time", "", ]
    for model in models:
        contrasts = metrics[model]["contrasts"]
        lines += [
            f"### {MODEL_LABELS.get(model, model)}",
            "",
            "| Contrast | Factor moved | Safe-approval change | 95% CI | Unsafe-approval change | Blind spots |",
            "|---|---|---:|---|---:|---:|",
        ]
        for name, _, _, _ in CONTRASTS:
            entry = contrasts.get(name, {})
            lines.append(
                f"| `{name}` | {entry.get('factor_moved')} | "
                f"{_pp(entry.get('safe_approval_delta'))} | {_ci(entry, 'safe_approval_delta')} | "
                f"{_pp(entry.get('unsafe_approval_delta'))} | "
                f"{_pct(entry.get('paired_blind_spot_rate'))} |"
            )
        lines.append("")

    lines += ["## Figures", ""]
    lines += [f"![{Path(path).stem}]({Path(path).name})" for path in figures]
    lines += ["", "## Stability", ""]
    stability = summary.get("stability", {})
    lines.append(
        f"Repeat cells: {stability.get('cells', 0)}, unanimous rate: "
        f"{_pct(stability.get('unanimous_rate'))}"
        if stability.get("available")
        else "No repeat data in this run."
    )
    lines.append("")
    if manifest:
        lines += [
            "## Provenance",
            "",
            f"- Traces: {manifest.get('trace_count')} · views: {manifest.get('view_count')}",
            f"- Prompt: {manifest.get('prompt_source')}",
            f"- Providers: {manifest.get('provider_source')}",
            f"- Commit: `{manifest.get('source_commit')}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def _read_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
