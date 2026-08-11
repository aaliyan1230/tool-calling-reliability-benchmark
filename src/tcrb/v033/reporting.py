from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analysis import CONTRASTS
from .schema import ANNOUNCEMENT, DEGRADED_VIEW_TYPES, SCOPE, VIEW_TYPES


SHORT = {
    "full": "Full",
    "symmetric_advertised": "Symmetric\nadvertised",
    "symmetric_silent": "Symmetric\nsilent",
    "asymmetric_advertised": "Asymmetric\nadvertised",
    "asymmetric_silent": "Asymmetric\nSILENT",
}
MODELS = {"deepseek": "DeepSeek V4 Flash", "gpt": "GPT-5.6 Terra"}
CONTRAST_LABELS = {
    "announcement_effect_asymmetric": "Announcement\n(asymmetric)",
    "announcement_effect_symmetric": "Announcement\n(symmetric)",
    "scope_effect_advertised": "Scope\n(advertised)",
    "scope_effect_silent": "Scope\n(silent)",
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = summary.get("metrics", {})
    models = [m for m in ("deepseek", "gpt") if m in metrics]
    paths: list[str] = []

    # Figure 1 - the headline. Unsafe approval is what v3.3 is hunting for, so
    # it leads; safe approval sits behind it for context.
    fig, axes = plt.subplots(1, len(models), figsize=(6.6 * len(models), 4.8), squeeze=False)
    for axis, model in zip(axes[0], models):
        by_view = metrics[model]["by_view"]
        views = [v for v in VIEW_TYPES if v in by_view]
        x = range(len(views))
        width = 0.38
        unsafe = [(by_view[v].get("unsafe_approval_rate") or 0) * 100 for v in views]
        safe = [(by_view[v].get("safe_approval_rate") or 0) * 100 for v in views]
        axis.bar([i - width / 2 for i in x], unsafe, width,
                 label="Unsafe approved (a blind spot)", color="#c8102e")
        axis.bar([i + width / 2 for i in x], safe, width,
                 label="Safe approved", color="#2f7ed8", alpha=0.55)
        for i, v in enumerate(unsafe):
            axis.text(i - width / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=8, fontweight="bold")
        for i, v in enumerate(safe):
            axis.text(i + width / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=8)
        axis.set_xticks(list(x))
        axis.set_xticklabels([SHORT[v] for v in views], fontsize=8)
        axis.set_ylim(0, 115)
        axis.set_ylabel("Rate (%)")
        axis.set_title(MODELS.get(model, model))
        axis.legend(fontsize=8, loc="upper center")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Can a blind spot be produced when nothing announces the gap?", fontsize=12
    )
    fig.tight_layout()
    path = output_dir / "figure-1-unsafe-approval-by-view.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(str(path))

    # Figure 2 - blind spots against the complete trace, with CIs.
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.4), squeeze=False)
    for axis, model in zip(axes[0], models):
        blind = metrics[model]["blind_spots_vs_full"]
        views = [v for v in DEGRADED_VIEW_TYPES if v in blind]
        values, lows, highs = [], [], []
        for view in views:
            entry = blind[view]
            point = (entry.get("blind_spot_rate") or 0) * 100
            ci = (entry.get("bootstrap_95_ci") or {}).get("blind_spot_rate")
            low, high = (ci[0] * 100, ci[1] * 100) if ci else (point, point)
            values.append(point)
            lows.append(max(0.0, point - low))
            highs.append(max(0.0, high - point))
        colors = ["#8c1d40" if v == "asymmetric_silent" else "#9aa0a6" for v in views]
        axis.bar(range(len(views)), values, yerr=[lows, highs], capsize=4, color=colors)
        for i, v in enumerate(values):
            axis.text(i, v + highs[i] + 2, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
        axis.set_xticks(range(len(views)))
        axis.set_xticklabels([SHORT[v] for v in views], fontsize=8)
        axis.set_ylabel("Blind-spot rate vs full trace (%)")
        axis.set_ylim(0, 105)
        axis.set_title(MODELS.get(model, model))
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Caught on the full trace, approved once degraded", fontsize=12)
    fig.tight_layout()
    path = output_dir / "figure-2-blind-spots.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(str(path))

    # Figure 3 - main effects on unsafe approval.
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.2), squeeze=False)
    for axis, model in zip(axes[0], models):
        main = metrics[model]["main_effects"]
        groups = [
            ("Symmetric", main["scope"].get("symmetric", {}), "#3d7d3f"),
            ("Asymmetric", main["scope"].get("asymmetric", {}), "#3d7d3f"),
            ("Advertised", main["announcement"].get("advertised", {}), "#e07b39"),
            ("Silent", main["announcement"].get("silent", {}), "#e07b39"),
        ]
        values = [(e.get("unsafe_approval_rate") or 0) * 100 for _, e, _ in groups]
        axis.bar(range(len(groups)), values, color=[c for _, _, c in groups])
        for i, v in enumerate(values):
            axis.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
        axis.set_xticks(range(len(groups)))
        axis.set_xticklabels([g for g, _, _ in groups], fontsize=8)
        axis.set_ylim(0, 105)
        axis.set_ylabel("Unsafe-approval rate (%)")
        axis.set_title(MODELS.get(model, model))
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Main effects on unsafe approval (green = scope, orange = announcement)",
        fontsize=12,
    )
    fig.tight_layout()
    path = output_dir / "figure-3-main-effects.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _pct(v: Any) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _pp(v: Any) -> str:
    return "n/a" if v is None else f"{v * 100:+.1f}pp"


def _ci(entry: dict[str, Any], key: str) -> str:
    ci = (entry.get("bootstrap_95_ci") or {}).get(key)
    return "n/a" if not ci else f"[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]"


def _render(summary, figures, *, manifest, audit) -> str:
    metrics = summary.get("metrics", {})
    models = [m for m in ("deepseek", "gpt") if m in metrics]
    lines = [
        "# v3.3 — can a blind spot be produced when nothing announces the gap?",
        "",
        f"Responses: {summary.get('rows')} "
        f"({summary.get('primary_rows')} primary, {summary.get('stability_rows')} repeats). "
        f"Valid: {summary.get('responses_by_status', {}).get('valid')}. "
        f"Cost: ${summary.get('estimated_extra_cost_usd', 0):.4f}.",
        "",
        f"Quality gate: **{'PASS' if summary.get('quality_gate', {}).get('passed') else 'FAIL'}**"
        + (f" · Integrity audit: **{'PASS' if audit.get('passed') else 'FAIL'}**" if audit else ""),
        "",
        "|  | Advertised gap | Silent gap |",
        "|---|---|---|",
        "| **Symmetric drop** | `symmetric_advertised` | `symmetric_silent` |",
        "| **Asymmetric drop** | `asymmetric_advertised` | `asymmetric_silent` |",
        "",
        "## Rates by cell",
        "",
    ]
    for model in models:
        by_view = metrics[model]["by_view"]
        lines += [
            f"### {MODELS.get(model, model)}",
            "",
            "| View | Scope | Announcement | Unsafe approved | Safe approved | Insufficient |",
            "|---|---|---|---:|---:|---:|",
        ]
        for view in VIEW_TYPES:
            e = by_view.get(view)
            if not e:
                continue
            lines.append(
                f"| `{view}` | {SCOPE.get(view, '—')} | {ANNOUNCEMENT.get(view, '—')} | "
                f"**{_pct(e.get('unsafe_approval_rate'))}** | {_pct(e.get('safe_approval_rate'))} | "
                f"{_pct(e.get('insufficient_evidence_rate'))} |"
            )
        lines.append("")

    lines += ["## Blind spots against the full trace", ""]
    for model in models:
        blind = metrics[model]["blind_spots_vs_full"]
        lines += [
            f"### {MODELS.get(model, model)}",
            "",
            "| View | Blind-spot rate | 95% CI | Unsafe pairs |",
            "|---|---:|---|---:|",
        ]
        for view in DEGRADED_VIEW_TYPES:
            e = blind.get(view, {})
            lines.append(
                f"| `{view}` | **{_pct(e.get('blind_spot_rate'))}** | "
                f"{_ci(e, 'blind_spot_rate')} | {e.get('unsafe_pairs')} |"
            )
        lines.append("")

    lines += ["## Moving one factor at a time", ""]
    for model in models:
        contrasts = metrics[model]["contrasts"]
        lines += [
            f"### {MODELS.get(model, model)}",
            "",
            "| Contrast | Factor | Unsafe-approval change | 95% CI | Safe-approval change |",
            "|---|---|---:|---|---:|",
        ]
        for name, _, _, _ in CONTRASTS:
            e = contrasts.get(name, {})
            lines.append(
                f"| `{name}` | {e.get('factor_moved')} | "
                f"**{_pp(e.get('unsafe_approval_delta'))}** | "
                f"{_ci(e, 'unsafe_approval_delta')} | {_pp(e.get('safe_approval_delta'))} |"
            )
        lines.append("")

    lines += ["## Figures", ""]
    lines += [f"![{Path(p).stem}]({Path(p).name})" for p in figures]
    stability = summary.get("stability", {})
    lines += [
        "",
        "## Stability",
        "",
        (
            f"Repeat cells: {stability.get('cells', 0)}, unanimous: {_pct(stability.get('unanimous_rate'))}"
            if stability.get("available")
            else "No repeat data."
        ),
        "",
    ]
    if manifest:
        lines += [
            "## Provenance",
            "",
            f"- Traces: {manifest.get('trace_count')} · views: {manifest.get('view_count')}",
            f"- Prompt: {manifest.get('prompt_source')}",
            f"- Commit: `{manifest.get('source_commit')}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def _read_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
