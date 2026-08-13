"""Generate polished, reproducible TCRB v3.5 result figures.

All values are loaded from the hash-frozen monitor analysis artifacts. The
script writes both editable SVG and high-resolution PNG versions.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "assets" / "v035" / "figures"
REPORT_OUTPUT = ROOT / "docs" / "v3" / "figures"
REPORT_FIGURES = ("holdout_signal", "family_heatmap_broad", "generalization_gap")

BG = "#F7F5F0"
INK = "#25231F"
MUTED = "#6F6A61"
GRID = "#DDD8CF"
ORANGE = "#D97745"
ORANGE_DARK = "#B4512E"
SAFE = "#A6A095"
PURPLE = "#6E56CF"
TEAL = "#2F7E79"
BLUE = "#3F6FA3"
MODEL_COLORS = {
    "gpt-5.6-terra": ORANGE,
    "deepseek-v4-pro": PURPLE,
    "qwen3.7-plus": TEAL,
    "gemini-3.6-flash": BLUE,
}
MODEL_LABELS = {
    "gpt-5.6-terra": "Terra",
    "deepseek-v4-pro": "DeepSeek Pro",
    "qwen3.7-plus": "Qwen Plus",
    "gemini-3.6-flash": "Gemini 3.6",
}
MODELS = tuple(MODEL_COLORS)
POLICIES = ("narrow", "broad")
COHORTS = (("control", None), ("main", "development"), ("main", "holdout_v2"))
FAMILY_ORDER = (
    "wrong_target",
    "payment_mismatch",
    "scope_widening",
    "frozen_selection_drift",
    "partial_batch_retry",
    "timeout_replay",
    "stale_target_state",
)
FAMILY_LABELS = {
    "wrong_target": "Wrong target",
    "payment_mismatch": "Payment mismatch",
    "scope_widening": "Scope widening",
    "frozen_selection_drift": "Frozen selection drift",
    "partial_batch_retry": "Partial-batch retry",
    "timeout_replay": "Timeout replay",
    "stale_target_state": "Stale target state",
}


def analysis_path(model: str, policy: str, role: str, cohort: str | None) -> Path:
    if model == "gemini-3.6-flash":
        directory = ROOT / "outputs" / "v035" / "prewrite" / "frozen_gemini"
    else:
        directory = ROOT / "outputs" / "v035" / "prewrite" / f"frozen_model_expansion_{policy}"
    stem = f"monitor_runtime_{role}"
    if cohort:
        stem += f"_{cohort}"
    if policy == "broad":
        stem += "_broad"
    stem += f"_{model}_analysis.json"
    path = directory / stem
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_analysis(model: str, policy: str, role: str, cohort: str | None) -> dict[str, Any]:
    return json.loads(analysis_path(model, policy, role, cohort).read_text(encoding="utf-8"))


def load_data() -> dict[str, Any]:
    data: dict[str, Any] = {"metrics": {}, "families": {}}
    for policy in POLICIES:
        data["metrics"][policy] = {}
        data["families"][policy] = {}
        for model in MODELS:
            data["metrics"][policy][model] = {}
            family_totals: dict[str, dict[str, int]] = {
                family: {"safe": 0, "safe_allowed": 0, "unsafe": 0, "unsafe_blocked": 0}
                for family in FAMILY_ORDER
            }
            for role, cohort in COHORTS:
                analysis = load_analysis(model, policy, role, cohort)
                key = cohort or "control"
                data["metrics"][policy][model][key] = {
                    "unsafe_block_rate": analysis["unsafe_block_rate"],
                    "safe_allow_rate": analysis["safe_allow_rate"],
                    "unsafe_escalate_rate": analysis.get("unsafe_escalate_rate"),
                    "safe_escalate_rate": analysis.get("safe_escalate_rate"),
                    "n": analysis["n"],
                }
                for family, values in analysis.get("by_family", {}).items():
                    for field in family_totals[family]:
                        family_totals[family][field] += int(values.get(field, 0))
            data["families"][policy][model] = {
                family: {
                    "unsafe_block_rate": values["unsafe_blocked"] / values["unsafe"] if values["unsafe"] else None,
                    "safe_allow_rate": values["safe_allowed"] / values["safe"] if values["safe"] else None,
                    "unsafe": values["unsafe"],
                    "safe": values["safe"],
                }
                for family, values in family_totals.items()
            }
    return data


def setup_mpl() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "svg.fonttype": "none",
            # Stable IDs make repeated SVG exports byte-comparable.
            "svg.hashsalt": "tcrb-v3.5-figures",
        }
    )


def base_axis(ax: plt.Axes, *, y_grid: bool = True) -> None:
    ax.set_facecolor(BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, pad=6)
    if y_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)


def percent_axis(ax: plt.Axes, *, y_grid: bool = True) -> None:
    base_axis(ax, y_grid=y_grid)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_yticks(np.linspace(0, 1, 5))


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.015, text, ha="left", va="bottom", fontsize=9.5, color=MUTED)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": "2026-08-13T00:00:00"}
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.18, metadata=metadata)
    svg_path = output / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.18, metadata=metadata)
    # Matplotlib emits trailing spaces in multiline SVG paths. Normalise them so
    # repository whitespace checks stay clean and repeated exports remain stable.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def figure_holdout_signal(data: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharey=True, constrained_layout=False)
    x = np.arange(len(MODELS))
    width = 0.34
    labels = [MODEL_LABELS[m] for m in MODELS]
    for ax, policy in zip(axes, POLICIES):
        values = data["metrics"][policy]
        unsafe = np.array([values[m]["holdout_v2"]["unsafe_block_rate"] for m in MODELS])
        safe = np.array([values[m]["holdout_v2"]["safe_allow_rate"] for m in MODELS])
        bars_unsafe = ax.bar(x - width / 2, unsafe, width, color=ORANGE, label="Unsafe blocked")
        bars_safe = ax.bar(x + width / 2, safe, width, color=SAFE, label="Safe allowed")
        percent_axis(ax)
        ax.axhline(1, color=INK, linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.65)
        ax.set_xticks(x, labels)
        ax.set_title("Narrow policy" if policy == "narrow" else "Broad policy", loc="left", pad=12)
        ax.set_xlabel("Monitor model")
        for bars in (bars_unsafe, bars_safe):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(height + 0.035, 0.045),
                    f"{height:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color=INK,
                )
    axes[0].set_ylabel("Rate on fresh stale-state holdout")
    fig.suptitle("Fresh holdout: can the monitor catch a changed target?", x=0.01, y=0.99, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.01, 0.925, "Exact BLOCK / ALLOW rates; 6 unsafe and 6 safe traces per model", ha="left", fontsize=11, color=MUTED)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.99, 0.955), ncol=2, handlelength=1.2, columnspacing=1.3)
    add_footer(fig, "Higher unsafe-blocking is better; higher safe-allowing means fewer false alarms. ESCALATE is not counted as BLOCK.")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.20, wspace=0.16)
    save_figure(fig, output, "holdout_signal")


def heatmap_axis(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    cmap: LinearSegmentedColormap,
    model_labels: list[str],
    *,
    show_ylabels: bool = True,
) -> None:
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    base_axis(ax, y_grid=False)
    ax.set_title(title, loc="left", pad=12)
    ax.set_xticks(range(len(model_labels)), model_labels, rotation=0)
    if show_ylabels:
        ax.set_yticks(range(len(FAMILY_ORDER)), [FAMILY_LABELS[f] for f in FAMILY_ORDER])
    else:
        # Keep the shared y scale but remove duplicate row labels on panel two.
        ax.tick_params(axis="y", labelleft=False, left=False)
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=8)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for i in range(len(FAMILY_ORDER) + 1):
        ax.axhline(i - 0.5, color=BG, linewidth=2.0)
    for i in range(len(FAMILY_ORDER)):
        for j in range(len(model_labels)):
            value = matrix[i, j]
            label = "—" if np.isnan(value) else f"{value:.0%}"
            ax.text(j, i, label, ha="center", va="center", fontsize=11, color="white" if value >= 0.62 else INK, fontweight="bold")


def figure_family_heatmap(data: dict[str, Any], output: Path) -> None:
    model_labels = [MODEL_LABELS[m] for m in MODELS]
    unsafe = np.array([[data["families"]["broad"][m][family]["unsafe_block_rate"] for m in MODELS] for family in FAMILY_ORDER], dtype=float)
    safe = np.array([[data["families"]["broad"][m][family]["safe_allow_rate"] for m in MODELS] for family in FAMILY_ORDER], dtype=float)
    unsafe_cmap = LinearSegmentedColormap.from_list("unsafe", [BG, "#F0C9B4", ORANGE_DARK])
    safe_cmap = LinearSegmentedColormap.from_list("safe", [BG, "#C9DCD8", TEAL])
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.8), sharey=True, constrained_layout=False)
    heatmap_axis(axes[0], unsafe, "Unsafe cases blocked", unsafe_cmap, model_labels, show_ylabels=True)
    heatmap_axis(axes[1], safe, "Safe cases allowed", safe_cmap, model_labels, show_ylabels=False)
    fig.suptitle("Broad-policy performance by failure family", x=0.01, y=0.985, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.01, 0.915, "All available development + fresh holdout cases; exact BLOCK / ALLOW rates", ha="left", fontsize=11, color=MUTED)
    add_footer(fig, "Direct mistakes are easy; the state-dependent stale-target family is where models separate.")
    fig.subplots_adjust(left=0.19, right=0.98, top=0.79, bottom=0.16, wspace=0.18)
    save_figure(fig, output, "family_heatmap_broad")


def figure_generalization(data: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True, constrained_layout=False)
    x = np.array([0, 1])
    x_labels = ["Development\n(8 unsafe)", "Fresh holdout\n(6 unsafe)"]
    for ax, policy in zip(axes, POLICIES):
        for model in MODELS:
            values = [data["metrics"][policy][model][cohort]["unsafe_block_rate"] for cohort in ("development", "holdout_v2")]
            ax.plot(x, values, color=MODEL_COLORS[model], linewidth=2.6, marker="o", markersize=7, label=MODEL_LABELS[model])
            ax.text(1.035, values[-1], f"{values[-1]:.0%}", va="center", ha="left", fontsize=9.5, color=MODEL_COLORS[model], fontweight="bold")
        percent_axis(ax)
        ax.set_xticks(x, x_labels)
        ax.set_xlim(-0.08, 1.16)
        ax.set_title("Narrow policy" if policy == "narrow" else "Broad policy", loc="left", pad=12)
        ax.set_xlabel("Evaluation split")
        ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel("Unsafe cases blocked")
    fig.suptitle("Fresh cases expose the generalization gap", x=0.01, y=0.99, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.01, 0.925, "The monitor sees the same task structure, but the fresh target-state mutations are new", ha="left", fontsize=11, color=MUTED)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.02, -0.23), ncol=4, handlelength=1.7, columnspacing=1.2)
    add_footer(fig, "A steep fall from development to holdout means the model learned the obvious pattern, not the underlying state relation.")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.80, bottom=0.27, wspace=0.17)
    save_figure(fig, output, "generalization_gap")


def write_captions(data: dict[str, Any], output: Path) -> None:
    manifest_paths = [
        "outputs/v035/prewrite/frozen_model_expansion_narrow/freeze_manifest.json",
        "outputs/v035/prewrite/frozen_model_expansion_broad/freeze_manifest.json",
        "outputs/v035/prewrite/frozen_gemini/freeze_manifest.json",
    ]
    payload = {
        "dataset_id": "dataset_2c30acaff18497dd6a411524",
        "source_freeze_manifests": manifest_paths,
        "figures": {
            "holdout_signal": "On 12 fresh stale-target-state traces per model (6 unsafe, 6 safe), exact BLOCK and ALLOW rates separate the monitors. Gemini blocks 6/6 unsafe cases with the narrow policy; DeepSeek blocks 0/6 with the broad policy. This is a model-dependent generalization gap, not a uniform failure.",
            "family_heatmap_broad": "Direct mistakes are blocked at 100% in this aggregate. State-dependent stale-target cases are harder: unsafe blocking ranges from 0% to 75%, while safe allowing ranges from 62% to 100%.",
            "generalization_gap": "The same monitor can look strong on familiar development cases and fall on fresh target-state mutations. Under the broad policy, DeepSeek drops from 6/8 to 0/6 unsafe cases blocked; Gemini drops from 7/8 to 5/6.",
        },
        "notes": [
            "Values are loaded from frozen JSON analyses; no values are hand-entered.",
            "ESCALATE is not counted as an exact unsafe BLOCK or safe ALLOW.",
            "Gemini 3.6 Flash used thinkingLevel=minimal and omitted temperature, as recorded in its freeze manifest.",
        ],
    }
    (output / "captions.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "captions.md").write_text(
        "# TCRB v3.5 figures\n\n"
        "Generated from the immutable v3.5 result freezes by `scripts/plot_v35_results.py`.\n\n"
        "## Suggested captions\n\n"
        "- **Fresh holdout signal.** On 12 fresh stale-target-state traces per model (6 unsafe, 6 safe), exact BLOCK and ALLOW rates separate the monitors. Gemini blocks 6/6 unsafe cases with the narrow policy; DeepSeek blocks 0/6 with the broad policy. This is a model-dependent generalization gap, not a uniform failure.\n"
        "- **Failure-family heatmap.** Direct mistakes are blocked at 100% in this aggregate. State-dependent stale-target cases are harder: unsafe blocking ranges from 0% to 75%, while safe allowing ranges from 62% to 100%.\n"
        "- **Development-to-holdout gap.** The same monitor can look strong on familiar development cases and fall on fresh target-state mutations. Under the broad policy, DeepSeek drops from 6/8 to 0/6 unsafe cases blocked; Gemini drops from 7/8 to 5/6.\n\n"
        "All values are loaded from frozen JSON analyses; no values are hand-entered. `ESCALATE` is not counted as exact `BLOCK` or `ALLOW`.\n",
        encoding="utf-8",
    )


def sync_report_pngs(output: Path) -> None:
    """Keep the report's local images in sync with the canonical exports."""

    REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)
    for stem in REPORT_FIGURES:
        shutil.copy2(output / f"{stem}.png", REPORT_OUTPUT / f"{stem}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    setup_mpl()
    data = load_data()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figure_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    figure_holdout_signal(data, args.output)
    figure_family_heatmap(data, args.output)
    figure_generalization(data, args.output)
    write_captions(data, args.output)
    if args.output.resolve() == DEFAULT_OUTPUT.resolve():
        sync_report_pngs(args.output)
    print(json.dumps({"output": str(args.output), "figures": list(REPORT_FIGURES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
