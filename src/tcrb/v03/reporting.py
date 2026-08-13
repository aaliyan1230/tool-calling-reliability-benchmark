from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROMPT_LABELS = {
    "baseline": "Baseline",
    "evidence_first": "Evidence-first",
}
PROVIDER_LABELS = {
    "deepseek": "DeepSeek V4 Flash",
    "gpt": "GPT-5.6 Terra",
}


def build_report(run_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    """Build reproducible figures and a short brief from scored pilot outputs."""
    summary_path = run_dir / "summary.json"
    scores_path = run_dir / "scores.jsonl"
    if not summary_path.exists() or not scores_path.exists():
        raise FileNotFoundError("run analyze before building the report")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scores = [
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    primary = [
        row
        for row in scores
        if row.get("repeat_index") == 0
        and not row.get("exploratory")
        and row.get("valid")
    ]
    if not primary:
        raise ValueError("no valid primary scores found")

    destination = output_dir or (run_dir / "report")
    destination.mkdir(parents=True, exist_ok=True)
    interaction_paths = _plot_interaction(primary, destination)
    comparison_paths = _plot_prompt_improvement(summary, destination)
    stability_paths: list[Path] = []
    if summary.get("stability", {}).get("available"):
        stability_paths = _plot_stability(summary["stability"], destination)

    captions_path = destination / "figure-captions.md"
    captions_path.write_text(
        _figure_captions(primary, summary, bool(stability_paths)), encoding="utf-8"
    )
    brief_path = destination / "pilot-brief.md"
    brief_path.write_text(_pilot_brief(summary), encoding="utf-8")
    outreach_path = destination / "monika-outreach-draft.md"
    outreach_path.write_text(_outreach_draft(), encoding="utf-8")

    paths = interaction_paths + comparison_paths + stability_paths + [
        captions_path,
        brief_path,
        outreach_path,
    ]
    return {
        "providers": sorted({row["provider"] for row in primary}),
        "primary_rows": len(primary),
        "artifacts": [str(path) for path in paths],
    }


def _plot_interaction(rows: list[dict[str, Any]], destination: Path) -> list[Path]:
    plt = _pyplot()
    providers = sorted({row["provider"] for row in rows})
    prompts = [name for name in PROMPT_LABELS if any(row["prompt_variant"] == name for row in rows)]
    figure, axes = plt.subplots(
        len(providers),
        len(prompts),
        figsize=(5.5 * len(prompts), 3.7 * len(providers) + 1.8),
        squeeze=False,
        sharey=True,
    )
    figure.patch.set_facecolor("#fbfaf7")
    colors = {"verified": "#1b7f79", "warning": "#df765e"}
    states = ("correct", "corrupt")
    state_labels = ("Correct payload", "Corrupted payload")
    width = 0.32

    for row_index, provider in enumerate(providers):
        for column_index, prompt in enumerate(prompts):
            axis = axes[row_index][column_index]
            subset = [
                row
                for row in rows
                if row["provider"] == provider and row["prompt_variant"] == prompt
            ]
            for label_index, self_label in enumerate(("verified", "warning")):
                values = [
                    _mean(
                        bool(row["flagged"])
                        for row in subset
                        if row["payload_state"] == state and row["self_label"] == self_label
                    )
                    for state in states
                ]
                positions = [index + (label_index - 0.5) * width for index in range(2)]
                bars = axis.bar(
                    positions,
                    values,
                    width=width,
                    color=colors[self_label],
                    label=self_label.title(),
                    zorder=3,
                )
                for bar, value in zip(bars, values, strict=True):
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(value + 0.035, 1.035),
                        _percent(value),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="#17202a",
                    )
            axis.set_title(
                f"{PROVIDER_LABELS.get(provider, provider)} · {PROMPT_LABELS[prompt]}",
                fontsize=12,
                fontweight="bold",
                loc="left",
                pad=10,
            )
            axis.set_xticks((0, 1), state_labels)
            axis.set_ylim(0, 1.12)
            axis.set_yticks((0, 0.25, 0.5, 0.75, 1), ("0%", "25%", "50%", "75%", "100%"))
            axis.grid(axis="y", color="#d8d5ce", linewidth=0.8, alpha=0.8, zorder=0)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#aaa69d")
            axis.set_facecolor("#fbfaf7")
            if column_index == 0:
                axis.set_ylabel("Cases flagged")
            else:
                axis.tick_params(axis="y", length=0)

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(0.97, 0.89))
    figure.suptitle(
        "Does the monitor follow evidence or the tool’s own label?",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#17202a",
    )
    figure.text(
        0.055,
        0.895,
        "Within each pair, the payload and external evidence are identical; only ‘verified’ vs ‘warning’ changes.",
        ha="left",
        fontsize=10,
        color="#5c625f",
    )
    figure.tight_layout(rect=(0.03, 0.04, 0.98, 0.78))
    return _save_figure(figure, destination / "figure-1-label-evidence-interaction")


def _plot_stability(stability: dict[str, Any], destination: Path) -> list[Path]:
    plt = _pyplot()
    condition_data = stability.get("by_condition", {})
    providers = sorted({key.split(":", 1)[0] for key in condition_data})
    metrics = (
        ("unanimous_decision_rate", "Same decision\nin all 3 runs"),
        ("false_alarm_rate", "False alarms"),
        ("self_label_effect", "Self-label effect"),
        ("corruption_detection_rate", "Corruption\ndetected"),
    )
    figure, axes = plt.subplots(
        len(providers),
        1,
        figsize=(9.5, 3.4 * len(providers) + 1.7),
        squeeze=False,
    )
    figure.patch.set_facecolor("#fbfaf7")
    colors = {"baseline": "#df765e", "evidence_first": "#1b7f79"}
    width = 0.34
    for row_index, provider in enumerate(providers):
        axis = axes[row_index][0]
        for prompt_index, prompt in enumerate(("baseline", "evidence_first")):
            item = condition_data.get(f"{provider}:{prompt}")
            if not item:
                continue
            majority = item.get("majority_vote_metrics", {})
            values = [
                item.get("unanimous_decision_rate"),
                majority.get("false_alarm_rate"),
                majority.get("self_label_effect"),
                majority.get("corruption_detection_rate"),
            ]
            positions = [index + (prompt_index - 0.5) * width for index in range(len(metrics))]
            bars = axis.bar(
                positions,
                values,
                width=width,
                color=colors[prompt],
                label=PROMPT_LABELS[prompt],
                zorder=3,
            )
            for bar, value in zip(bars, values, strict=True):
                if value is None:
                    continue
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(value + 0.035, 1.035),
                    _percent(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#17202a",
                )
        axis.set_title(PROVIDER_LABELS.get(provider, provider), loc="left", fontsize=12, fontweight="bold")
        axis.set_xticks(range(len(metrics)), [label for _, label in metrics])
        axis.set_ylim(0, 1.12)
        axis.set_yticks((0, 0.25, 0.5, 0.75, 1), ("0%", "25%", "50%", "75%", "100%"))
        axis.set_ylabel("Rate across repeated cases")
        axis.grid(axis="y", color="#d8d5ce", linewidth=0.8, alpha=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#aaa69d")
        axis.set_facecolor("#fbfaf7")

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(0.97, 0.89))
    figure.suptitle(
        "Evidence-first decisions were more repeatable",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#17202a",
    )
    figure.text(
        0.06,
        0.895,
        "Four fixed base cases, one per domain; each matched condition was run three times.",
        ha="left",
        fontsize=10,
        color="#5c625f",
    )
    figure.tight_layout(rect=(0.04, 0.04, 0.98, 0.78))
    return _save_figure(figure, destination / "figure-3-repeatability")


def _plot_prompt_improvement(summary: dict[str, Any], destination: Path) -> list[Path]:
    plt = _pyplot()
    comparisons = summary.get("prompt_comparison", {})
    providers = sorted(comparisons)
    if not providers:
        return []
    metric_specs = (
        ("self_label_effect_delta", "Less self-label influence", -1),
        ("false_alarm_rate_delta", "Fewer false alarms", -1),
        ("corruption_detection_rate_delta", "More corruption detected", 1),
        ("decision_accuracy_delta", "Higher decision accuracy", 1),
    )
    colors = {"deepseek": "#1b7f79", "gpt": "#6d5ba7"}
    markers = {"deepseek": "o", "gpt": "s"}
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    figure.patch.set_facecolor("#fbfaf7")
    axis.set_facecolor("#fbfaf7")
    offsets = _centered_offsets(len(providers), 0.16)
    all_bounds = [0.0]
    for provider_index, provider in enumerate(providers):
        comparison = comparisons[provider]
        intervals = comparison.get("bootstrap_95_ci", {})
        for metric_index, (key, _, direction) in enumerate(metric_specs):
            raw_value = comparison.get(key)
            raw_interval = intervals.get(key)
            if raw_value is None or not raw_interval:
                continue
            value = float(raw_value) * direction * 100
            bounds = sorted(float(bound) * direction * 100 for bound in raw_interval)
            y = metric_index + offsets[provider_index]
            color = colors.get(provider, "#34495e")
            axis.errorbar(
                value,
                y,
                xerr=[[value - bounds[0]], [bounds[1] - value]],
                fmt=markers.get(provider, "o"),
                markersize=7,
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=4,
                label=PROVIDER_LABELS.get(provider, provider) if metric_index == 0 else None,
                zorder=3,
            )
            axis.text(
                bounds[1] + 0.8,
                y,
                f"{value:+.1f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#17202a",
            )
            all_bounds.extend(bounds)
    axis.axvline(0, color="#7d827e", linewidth=1.2, zorder=1)
    minimum, maximum = min(all_bounds), max(all_bounds)
    padding = max(4.0, (maximum - minimum) * 0.18)
    axis.set_xlim(minimum - padding, maximum + padding)
    axis.set_yticks(range(len(metric_specs)), [label for _, label, _ in metric_specs])
    axis.invert_yaxis()
    axis.set_xlabel("Improvement from evidence-first prompt (percentage points)")
    axis.grid(axis="x", color="#d8d5ce", linewidth=0.8, alpha=0.8, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#aaa69d")
    axis.tick_params(axis="y", length=0)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=False,
        ncol=len(providers),
    )
    figure.suptitle(
        "Evidence-first reduced false alarms without hiding corruption",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#17202a",
    )
    figure.text(
        0.06,
        0.905,
        "Paired changes with 95% bootstrap intervals over 16 base cases. Right is better.",
        ha="left",
        fontsize=10,
        color="#5c625f",
    )
    figure.tight_layout(rect=(0.04, 0.14, 0.98, 0.83))
    return _save_figure(figure, destination / "figure-2-prompt-improvement")


def _figure_captions(
    rows: list[dict[str, Any]], summary: dict[str, Any], has_stability: bool
) -> str:
    providers = sorted({row["provider"] for row in rows})
    lines = [
        "# Figure captions",
        "",
        "## Figure 1 — Label/evidence interaction",
        "",
        "Within each pair, the payload and independent evidence are unchanged; only the visible self-label changes.",
        "",
    ]
    for provider in providers:
        provider_rows = [row for row in rows if row["provider"] == provider]
        provider_name = PROVIDER_LABELS.get(provider, provider)
        n_cases = len({row["base_case_id"] for row in provider_rows})
        facts: list[str] = []
        for prompt in ("baseline", "evidence_first"):
            subset = [row for row in provider_rows if row["prompt_variant"] == prompt]
            if not subset:
                continue
            clean_verified = _condition_rate(subset, "correct", "verified")
            clean_warning = _condition_rate(subset, "correct", "warning")
            facts.append(
                f"{PROMPT_LABELS[prompt]} clean cases: {_percent(clean_verified)} flagged with "
                f"`verified` versus {_percent(clean_warning)} with `warning`"
            )
        lines.append(
            f"- **{provider_name} ({n_cases} base cases):** "
            + "; ".join(facts)
            + "."
        )
    lines.extend(
        [
            "",
            "Corrupted cases provide the safety check: both prompts should keep flagging them.",
            "",
        ]
    )
    if has_stability:
        lines.extend(
            [
                "## Figure 2 — Paired prompt improvement",
                "",
                "Points are oriented so positive values are improvements. Whiskers are 95% bootstrap intervals "
                "from resampling the 16 base cases while keeping each prompt pair together. Both false-alarm "
                "reductions point in the desired direction, but their intervals touch zero; this pilot is too "
                "small for a firm population claim.",
                "",
                "## Figure 3 — Repeatability",
                "",
                "Four pre-selected base cases—one per domain—were each evaluated three times per matched "
                "condition. The chart reports decision unanimity and majority-vote performance. This is a "
                "small stability check, not a confidence interval or a new independently sampled test set.",
                "",
            ]
        )
    lines.extend(
        [
            "## Shared caveat",
            "",
            "These are deterministic simulated traces, not production logs. The core pilot contains 16 base "
            "cases, so small percentage changes can represent only one or two cases.",
            "",
        ]
    )
    return "\n".join(lines)


def _pilot_brief(summary: dict[str, Any]) -> str:
    providers = sorted({key.split(":", 1)[0] for key in summary.get("metrics", {})})
    complete_two_model = {"deepseek", "gpt"}.issubset(providers)
    lines = [
        "# Pilot brief: Do monitors trust a tool’s confidence label?",
        "",
        "## Question",
        "",
        "When a tool calls its own answer `verified`, does an AI monitor trust that label over independent records?",
        "",
        "## Test",
        "",
        "We built 16 realistic simulated cases across customer support, ecommerce, fintech, and developer tools. "
        "Each case was shown in four matched forms: correct/corrupt output × verified/warning label. Only one "
        "factor changed at a time. We compared a general baseline prompt with an evidence-first prompt that "
        "explicitly prioritizes independent records.",
        "",
        "## Results",
        "",
        "| Model | Prompt | Self-label effect | False alarms | Corruption detected | Accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for provider in providers:
        for prompt in ("baseline", "evidence_first"):
            metric = summary["metrics"].get(f"{provider}:{prompt}")
            if not metric:
                continue
            lines.append(
                f"| {PROVIDER_LABELS.get(provider, provider)} | {PROMPT_LABELS[prompt]} | "
                f"{_pp(metric.get('self_label_effect'))} | {_percent(metric.get('false_alarm_rate'))} | "
                f"{_percent(metric.get('corruption_detection_rate'))} | "
                f"{_percent(metric.get('decision_accuracy'))} |"
            )
    lines.extend(["", "## What this means", ""])
    if "deepseek" in providers:
        baseline = summary["metrics"].get("deepseek:baseline", {})
        evidence = summary["metrics"].get("deepseek:evidence_first", {})
        reduction = _relative_reduction(
            baseline.get("self_label_effect"), evidence.get("self_label_effect")
        )
        lines.append(
            "For DeepSeek, the evidence-first prompt reduced the measured label effect by "
            f"{_percent(reduction)} and false alarms by "
            f"{_points(baseline.get('false_alarm_rate'), evidence.get('false_alarm_rate'))}, while corruption "
            "detection stayed unchanged."
        )
    if complete_two_model:
        baseline = summary["metrics"].get("gpt:baseline", {})
        evidence = summary["metrics"].get("gpt:evidence_first", {})
        lines.append(
            "For GPT-5.6 Terra, evidence-first reduced false alarms by "
            f"{_points(baseline.get('false_alarm_rate'), evidence.get('false_alarm_rate'))} and improved "
            f"accuracy by {_points(evidence.get('decision_accuracy'), baseline.get('decision_accuracy'))}; "
            "corruption detection again stayed at 100%."
        )
        stability = summary.get("stability", {}).get("by_condition", {})
        lines.append(
            "On the four-case repeat check, decision unanimity changed from "
            f"{_percent(stability.get('deepseek:baseline', {}).get('unanimous_decision_rate'))} to "
            f"{_percent(stability.get('deepseek:evidence_first', {}).get('unanimous_decision_rate'))} for "
            "DeepSeek and from "
            f"{_percent(stability.get('gpt:baseline', {}).get('unanimous_decision_rate'))} to "
            f"{_percent(stability.get('gpt:evidence_first', {}).get('unanimous_decision_rate'))} for GPT."
        )
    else:
        lines.append(
            "This is still an interim one-model result. The locked GPT-5.6 Terra run is required before external outreach."
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "The traces are simulated and the 16-case sample is small. Both baseline label-effect confidence "
            "intervals include zero. Two developer-tool cases also showed that `allow` can be confused with "
            "approving a failed build rather than accepting a correct status report.",
            "",
            "## Next decision",
            "",
            (
                "Use the same fixed cases and prompts for GPT-5.6 Terra, then report both models without changing "
                "the success rule."
                if not complete_two_model
                else "The pre-registered contact gate did not pass. Share this as an honest early result about "
                "model-specific label sensitivity and a promising evidence-first mitigation—not as proof of a "
                "broad failure mode."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _outreach_draft() -> str:
    return """# Draft message to Monika

Hi Monika,

I’ve been running a small pilot on evidence provenance in agent monitoring. We built 16 matched simulated cases across four tool-use domains, varying whether an output was correct or corrupted and whether the tool called itself “verified” or warned it might be unreliable.

The label effect was model-specific: 9.4 points for DeepSeek V4 Flash and 3.1 for GPT-5.6 Terra, with wide intervals. Our pre-set gate for a broad label-trust effect did not pass. The more consistent result was that a short evidence-first instruction cut false alarms from 15.6% to 3.1% on DeepSeek and 9.4% to 0% on GPT, while corruption detection stayed at 100%. It also improved repeatability on a four-case rerun.

I’d really value your take on two questions: Is provenance-sensitive monitoring useful to pursue beyond prompting, and what realistic setting would make the strongest next test?

Happy to send the one-page brief and exact trajectories.
"""


def _condition_rate(rows: list[dict[str, Any]], state: str, label: str) -> float | None:
    return _mean(
        bool(row["flagged"])
        for row in rows
        if row["payload_state"] == state and row["self_label"] == label
    )


def _mean(values: Any) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _relative_reduction(before: float | None, after: float | None) -> float | None:
    if before in (None, 0) or after is None:
        return None
    return (before - after) / abs(before)


def _points(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "n/a"
    return f"{(before - after) * 100:.1f} percentage points"


def _percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _pp(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f} pp"


def _centered_offsets(count: int, spacing: float) -> list[float]:
    midpoint = (count - 1) / 2
    return [(index - midpoint) * spacing for index in range(count)]


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("reporting requires the dev extra: pip install -e '.[dev]'") from exc
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#2b3432",
            "xtick.color": "#4f5653",
            "ytick.color": "#4f5653",
        }
    )
    return plt


def _save_figure(figure: Any, stem: Path) -> list[Path]:
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    figure.savefig(png_path, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(svg_path, bbox_inches="tight", facecolor=figure.get_facecolor())
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    figure.clear()
    return [png_path, svg_path]
